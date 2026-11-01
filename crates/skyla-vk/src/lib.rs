//! Minimal Vulkan compute layer: one device, one compute queue, storage buffers,
//! specialized compute pipelines and timestamped dispatches.
//!
//! Drop order matters: buffers and pipelines must be dropped before the [`Context`].

use anyhow::{anyhow, bail, ensure, Context as _, Result};
use ash::vk;
use std::ffi::CStr;

pub use ash::vk::{BufferUsageFlags, MemoryPropertyFlags};

#[derive(Debug, Clone, Copy, Default)]
pub struct Caps {
    pub shader_float16: bool,
    pub shader_int8: bool,
    pub storage_buffer_16bit: bool,
    pub storage_buffer_8bit: bool,
}

pub struct Context {
    _entry: ash::Entry,
    instance: ash::Instance,
    pub physical: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub queue_family: u32,
    pub props: vk::PhysicalDeviceProperties,
    pub mem_props: vk::PhysicalDeviceMemoryProperties,
    pub subgroup_size: u32,
    pub max_allocation_size: u64,
    pub caps: Caps,
    cmd_pool: vk::CommandPool,
}

impl Context {
    /// Picks the first integrated GPU (or `SKYLA_DEVICE=<index>`) with a compute queue.
    /// Set `SKYLA_VALIDATION=1` to enable the Khronos validation layer.
    pub fn new() -> Result<Self> {
        let entry = unsafe { ash::Entry::load() }.context("loading vulkan-1.dll")?;

        let app = vk::ApplicationInfo::default()
            .application_name(c"skyla")
            .engine_name(c"skyla-kernel")
            .api_version(vk::API_VERSION_1_3);
        let layers = [c"VK_LAYER_KHRONOS_validation".as_ptr()];
        let validation = std::env::var("SKYLA_VALIDATION").is_ok_and(|v| v == "1");
        let mut create = vk::InstanceCreateInfo::default().application_info(&app);
        if validation {
            create = create.enabled_layer_names(&layers);
        }
        let instance =
            unsafe { entry.create_instance(&create, None) }.context("vkCreateInstance")?;

        let devices = unsafe { instance.enumerate_physical_devices() }?;
        let physical = match std::env::var("SKYLA_DEVICE") {
            Ok(i) => *devices.get(i.parse::<usize>()?).ok_or_else(|| {
                anyhow!("SKYLA_DEVICE={i} out of range ({} devices)", devices.len())
            })?,
            Err(_) => devices
                .iter()
                .copied()
                .find(|&d| {
                    unsafe { instance.get_physical_device_properties(d) }.device_type
                        == vk::PhysicalDeviceType::INTEGRATED_GPU
                })
                .or_else(|| devices.first().copied())
                .ok_or_else(|| anyhow!("no Vulkan devices found"))?,
        };

        let props = unsafe { instance.get_physical_device_properties(physical) };
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical) };

        let mut v11_props = vk::PhysicalDeviceVulkan11Properties::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut v11_props);
        unsafe { instance.get_physical_device_properties2(physical, &mut props2) };
        let subgroup_size = v11_props.subgroup_size;
        let max_allocation_size = v11_props.max_memory_allocation_size;

        let mut f11 = vk::PhysicalDeviceVulkan11Features::default();
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut f11)
            .push_next(&mut f12)
            .push_next(&mut f13);
        unsafe { instance.get_physical_device_features2(physical, &mut features2) };
        // glslc targeting Vulkan 1.3 emits LocalSizeId for `local_size_x_id`, which needs maintenance4.
        ensure!(
            f13.maintenance4 == vk::TRUE,
            "device lacks Vulkan 1.3 maintenance4"
        );
        let caps = Caps {
            shader_float16: f12.shader_float16 == vk::TRUE,
            shader_int8: f12.shader_int8 == vk::TRUE,
            storage_buffer_16bit: f11.storage_buffer16_bit_access == vk::TRUE,
            storage_buffer_8bit: f12.storage_buffer8_bit_access == vk::TRUE,
        };

        let queue_family =
            unsafe { instance.get_physical_device_queue_family_properties(physical) }
                .iter()
                .enumerate()
                // Prefer a compute queue with timestamps; graphics+compute is fine on an iGPU.
                .filter(|(_, q)| {
                    q.queue_flags.contains(vk::QueueFlags::COMPUTE) && q.timestamp_valid_bits > 0
                })
                .map(|(i, _)| i as u32)
                .next()
                .ok_or_else(|| anyhow!("no compute queue with timestamp support"))?;

        // Enable only what the device reports, so `probe` still works on weaker hardware.
        let mut en11 = vk::PhysicalDeviceVulkan11Features::default()
            .storage_buffer16_bit_access(caps.storage_buffer_16bit);
        let mut en12 = vk::PhysicalDeviceVulkan12Features::default()
            .shader_float16(caps.shader_float16)
            .shader_int8(caps.shader_int8)
            .storage_buffer8_bit_access(caps.storage_buffer_8bit);
        let mut en13 = vk::PhysicalDeviceVulkan13Features::default().maintenance4(true);
        let priorities = [1.0f32];
        let queue_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities)];
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .push_next(&mut en11)
            .push_next(&mut en12)
            .push_next(&mut en13);
        let device = unsafe { instance.create_device(physical, &device_info, None) }
            .context("vkCreateDevice")?;
        let queue = unsafe { device.get_device_queue(queue_family, 0) };

        let cmd_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }?;

        Ok(Self {
            _entry: entry,
            instance,
            physical,
            device,
            queue,
            queue_family,
            props,
            mem_props,
            subgroup_size,
            max_allocation_size,
            caps,
            cmd_pool,
        })
    }

    pub fn device_name(&self) -> String {
        unsafe { CStr::from_ptr(self.props.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    }

    pub fn memory_types(&self) -> &[vk::MemoryType] {
        &self.mem_props.memory_types[..self.mem_props.memory_type_count as usize]
    }

    pub fn memory_heaps(&self) -> &[vk::MemoryHeap] {
        &self.mem_props.memory_heaps[..self.mem_props.memory_heap_count as usize]
    }

    /// First memory type that has all of `required` and none of `excluded`.
    pub fn find_memory_type(
        &self,
        required: MemoryPropertyFlags,
        excluded: MemoryPropertyFlags,
    ) -> Option<u32> {
        self.memory_types()
            .iter()
            .position(|t| {
                t.property_flags.contains(required) && !t.property_flags.intersects(excluded)
            })
            .map(|i| i as u32)
    }

    pub fn alloc_buffer(
        &self,
        size: u64,
        usage: BufferUsageFlags,
        memory_type: u32,
    ) -> Result<Buffer> {
        let device = &self.device;
        let buffer = unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
        }?;
        let req = unsafe { device.get_buffer_memory_requirements(buffer) };
        if req.memory_type_bits & (1 << memory_type) == 0 {
            unsafe { device.destroy_buffer(buffer, None) };
            bail!("memory type {memory_type} not allowed for this buffer");
        }
        let memory = match unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(memory_type),
                None,
            )
        } {
            Ok(m) => m,
            Err(e) => {
                unsafe { device.destroy_buffer(buffer, None) };
                return Err(anyhow!(
                    "allocating {size} bytes in memory type {memory_type}: {e}"
                ));
            }
        };
        unsafe { device.bind_buffer_memory(buffer, memory, 0) }?;

        let host_visible = self.memory_types()[memory_type as usize]
            .property_flags
            .contains(MemoryPropertyFlags::HOST_VISIBLE);
        let mapped = if host_visible {
            Some(unsafe {
                device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            }? as *mut u8)
        } else {
            None
        };

        Ok(Buffer {
            device: device.clone(),
            buffer,
            memory,
            size,
            mapped,
        })
    }

    /// Builds a compute pipeline whose shader reads `n_buffers` storage buffers at bindings
    /// `0..n_buffers`. `spec` fills specialization constants `0..spec.len()` (all 32-bit).
    pub fn create_pipeline(
        &self,
        spirv: &[u8],
        n_buffers: u32,
        push_size: u32,
        spec: &[u32],
    ) -> Result<Pipeline> {
        let device = &self.device;
        let code = ash::util::read_spv(&mut std::io::Cursor::new(spirv))?;
        let module = unsafe {
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)
        }?;

        let bindings: Vec<_> = (0..n_buffers)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        }?;

        let push_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(push_size)];
        let set_layouts = [set_layout];
        let mut layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        if push_size > 0 {
            layout_info = layout_info.push_constant_ranges(&push_ranges);
        }
        let layout = unsafe { device.create_pipeline_layout(&layout_info, None) }?;

        let entries: Vec<_> = (0..spec.len() as u32)
            .map(|i| {
                vk::SpecializationMapEntry::default()
                    .constant_id(i)
                    .offset(i * 4)
                    .size(4)
            })
            .collect();
        let spec_data: Vec<u8> = spec.iter().flat_map(|v| v.to_le_bytes()).collect();
        let spec_info = vk::SpecializationInfo::default()
            .map_entries(&entries)
            .data(&spec_data);
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"main")
            .specialization_info(&spec_info);
        let pipeline = unsafe {
            device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(stage)
                    .layout(layout)],
                None,
            )
        }
        .map_err(|(_, e)| anyhow!("vkCreateComputePipelines: {e}"))?[0];
        unsafe { device.destroy_shader_module(module, None) };

        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(n_buffers.max(1))];
        let pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&pool_sizes),
                None,
            )
        }?;
        let set = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&set_layouts),
            )
        }?[0];

        Ok(Pipeline {
            device: device.clone(),
            pipeline,
            layout,
            set_layout,
            pool,
            set,
        })
    }

    /// Records `record` into a one-shot command buffer, submits it and waits.
    pub fn submit_and_wait(&self, record: impl FnOnce(vk::CommandBuffer)) -> Result<()> {
        let device = &self.device;
        let cmd = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }?[0];
        unsafe {
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            record(cmd);
            device.end_command_buffer(cmd)?;
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let cmds = [cmd];
            let result = device
                .queue_submit(
                    self.queue,
                    &[vk::SubmitInfo::default().command_buffers(&cmds)],
                    fence,
                )
                .and_then(|_| device.wait_for_fences(&[fence], true, u64::MAX));
            device.destroy_fence(fence, None);
            device.free_command_buffers(self.cmd_pool, &cmds);
            result?;
        }
        Ok(())
    }

    /// Dispatches `pipeline` `reps` times, serialized by barriers, and returns each
    /// dispatch's GPU time in nanoseconds.
    pub fn time_dispatch(
        &self,
        pipeline: &Pipeline,
        groups: [u32; 3],
        push: &[u8],
        reps: u32,
    ) -> Result<Vec<f64>> {
        let device = &self.device;
        let pool = unsafe {
            device.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(reps * 2),
                None,
            )
        }?;
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);

        let result = self.submit_and_wait(|cmd| unsafe {
            device.cmd_reset_query_pool(cmd, pool, 0, reps * 2);
            pipeline.bind(cmd);
            if !push.is_empty() {
                device.cmd_push_constants(
                    cmd,
                    pipeline.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push,
                );
            }
            for r in 0..reps {
                device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, pool, r * 2);
                device.cmd_dispatch(cmd, groups[0], groups[1], groups[2]);
                device.cmd_write_timestamp(
                    cmd,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    pool,
                    r * 2 + 1,
                );
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[barrier],
                    &[],
                    &[],
                );
            }
        });

        let mut stamps = vec![0u64; (reps * 2) as usize];
        let read = result.and_then(|_| {
            unsafe {
                device.get_query_pool_results(
                    pool,
                    0,
                    &mut stamps,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )
            }
            .map_err(Into::into)
        });
        unsafe { device.destroy_query_pool(pool, None) };
        read?;

        let period = self.props.limits.timestamp_period as f64;
        Ok(stamps
            .as_chunks::<2>()
            .0
            .iter()
            .map(|s| (s[1].wrapping_sub(s[0])) as f64 * period)
            .collect())
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_command_pool(self.cmd_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

pub struct Buffer {
    device: ash::Device,
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: u64,
    mapped: Option<*mut u8>,
}

impl Buffer {
    /// Host view of the buffer, if its memory type is host-visible.
    pub fn as_mut_slice(&mut self) -> Option<&mut [u8]> {
        self.mapped
            .map(|p| unsafe { std::slice::from_raw_parts_mut(p, self.size as usize) })
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            if self.mapped.is_some() {
                self.device.unmap_memory(self.memory);
            }
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

pub struct Pipeline {
    device: ash::Device,
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
}

impl Pipeline {
    /// Points bindings `0..buffers.len()` at `buffers`.
    pub fn bind_buffers(&self, buffers: &[&Buffer]) {
        let infos: Vec<[vk::DescriptorBufferInfo; 1]> = buffers
            .iter()
            .map(|b| {
                [vk::DescriptorBufferInfo::default()
                    .buffer(b.buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)]
            })
            .collect();
        let writes: Vec<_> = infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(self.set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(info)
            })
            .collect();
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };
    }

    pub fn bind(&self, cmd: vk::CommandBuffer) {
        unsafe {
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[self.set],
                &[],
            );
        }
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.layout, None);
            self.device.destroy_descriptor_pool(self.pool, None);
            self.device
                .destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}
