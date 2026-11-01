use anyhow::{anyhow, ensure, Context as _, Result};
use ash::vk;
use std::ffi::CStr;

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
    pub(crate) cmd_pool: vk::CommandPool,
}

fn pick_device(instance: &ash::Instance) -> Result<vk::PhysicalDevice> {
    let devices = unsafe { instance.enumerate_physical_devices() }?;
    if let Ok(i) = std::env::var("SKYLA_DEVICE") {
        return devices
            .get(i.parse::<usize>()?)
            .copied()
            .ok_or_else(|| anyhow!("SKYLA_DEVICE={i} out of range ({} devices)", devices.len()));
    }
    devices
        .iter()
        .copied()
        .find(|&d| {
            unsafe { instance.get_physical_device_properties(d) }.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU
        })
        .or_else(|| devices.first().copied())
        .ok_or_else(|| anyhow!("no Vulkan devices found"))
}

fn query_caps(instance: &ash::Instance, physical: vk::PhysicalDevice) -> Result<Caps> {
    let mut f11 = vk::PhysicalDeviceVulkan11Features::default();
    let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
    let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
    let mut features2 = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut f11)
        .push_next(&mut f12)
        .push_next(&mut f13);
    unsafe { instance.get_physical_device_features2(physical, &mut features2) };
    ensure!(f13.maintenance4 == vk::TRUE, "device lacks Vulkan 1.3 maintenance4");
    Ok(Caps {
        shader_float16: f12.shader_float16 == vk::TRUE,
        shader_int8: f12.shader_int8 == vk::TRUE,
        storage_buffer_16bit: f11.storage_buffer16_bit_access == vk::TRUE,
        storage_buffer_8bit: f12.storage_buffer8_bit_access == vk::TRUE,
    })
}

impl Context {
    pub fn new() -> Result<Self> {
        let entry = unsafe { ash::Entry::load() }.context("loading vulkan-1.dll")?;
        let app = vk::ApplicationInfo::default()
            .application_name(c"skyla")
            .engine_name(c"skyla-kernel")
            .api_version(vk::API_VERSION_1_3);
        let layers = [c"VK_LAYER_KHRONOS_validation".as_ptr()];
        let mut create = vk::InstanceCreateInfo::default().application_info(&app);
        if std::env::var("SKYLA_VALIDATION").is_ok_and(|v| v == "1") {
            create = create.enabled_layer_names(&layers);
        }
        let instance = unsafe { entry.create_instance(&create, None) }.context("vkCreateInstance")?;

        let physical = pick_device(&instance)?;
        let props = unsafe { instance.get_physical_device_properties(physical) };
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical) };
        let mut v11 = vk::PhysicalDeviceVulkan11Properties::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut v11);
        unsafe { instance.get_physical_device_properties2(physical, &mut props2) };
        let caps = query_caps(&instance, physical)?;

        let queue_family = unsafe { instance.get_physical_device_queue_family_properties(physical) }
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE) && q.timestamp_valid_bits > 0)
            .ok_or_else(|| anyhow!("no compute queue with timestamp support"))? as u32;

        let mut en11 =
            vk::PhysicalDeviceVulkan11Features::default().storage_buffer16_bit_access(caps.storage_buffer_16bit);
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
        let device = unsafe { instance.create_device(physical, &device_info, None) }.context("vkCreateDevice")?;
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
            subgroup_size: v11.subgroup_size,
            max_allocation_size: v11.max_memory_allocation_size,
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
