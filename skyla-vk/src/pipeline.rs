use crate::{Buffer, Context};
use anyhow::{anyhow, Result};
use ash::vk;
use skyla_shaders::Shader;

pub struct Pipeline {
    device: ash::Device,
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
}

impl Context {
    pub fn create_pipeline(&self, shader: &Shader, n_buffers: u32, push_size: u32, spec: &[u32]) -> Result<Pipeline> {
        self.create_pipeline_sets(shader, n_buffers, push_size, spec, 1)
    }

    pub fn create_pipeline_sets(
        &self,
        shader: &Shader,
        n_buffers: u32,
        push_size: u32,
        spec: &[u32],
        n_sets: u32,
    ) -> Result<Pipeline> {
        let device = &self.device;
        let code = ash::util::read_spv(&mut std::io::Cursor::new(shader.spirv))?;
        let module_info = vk::ShaderModuleCreateInfo::default().code(&code);
        let module = unsafe { device.create_shader_module(&module_info, None) }?;

        let bindings: Vec<_> = (0..n_buffers)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let set_layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let set_layout = unsafe { device.create_descriptor_set_layout(&set_layout_info, None) }?;

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
        let spec_info = vk::SpecializationInfo::default().map_entries(&entries).data(&spec_data);
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"main")
            .specialization_info(&spec_info);
        let create = [vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)];
        let pipeline = unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &create, None) }
            .map_err(|(_, e)| anyhow!("vkCreateComputePipelines: {e}"))?[0];
        unsafe { device.destroy_shader_module(module, None) };

        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(n_buffers.max(1) * n_sets)];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(n_sets)
            .pool_sizes(&pool_sizes);
        let pool = unsafe { device.create_descriptor_pool(&pool_info, None) }?;
        let layouts = vec![set_layout; n_sets as usize];
        let alloc = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);
        let sets = unsafe { device.allocate_descriptor_sets(&alloc) }?;

        Ok(Pipeline {
            device: device.clone(),
            pipeline,
            layout,
            set_layout,
            pool,
            sets,
        })
    }
}

impl Pipeline {
    pub fn bind_buffers(&self, buffers: &[&Buffer]) {
        self.bind_set_buffers(0, buffers);
    }

    pub fn bind_set_buffers(&self, set: usize, buffers: &[&Buffer]) {
        let infos: Vec<[vk::DescriptorBufferInfo; 1]> = buffers
            .iter()
            .map(|b| {
                [vk::DescriptorBufferInfo::default()
                    .buffer(b.buffer)
                    .range(vk::WHOLE_SIZE)]
            })
            .collect();
        let writes: Vec<_> = infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(self.sets[set])
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(info)
            })
            .collect();
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };
    }

    pub fn bind(&self, cmd: vk::CommandBuffer, set: usize) {
        let point = vk::PipelineBindPoint::COMPUTE;
        unsafe {
            self.device.cmd_bind_pipeline(cmd, point, self.pipeline);
            self.device
                .cmd_bind_descriptor_sets(cmd, point, self.layout, 0, &[self.sets[set]], &[]);
        }
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.layout, None);
            self.device.destroy_descriptor_pool(self.pool, None);
            self.device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}
