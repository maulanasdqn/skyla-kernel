use crate::{Context, Pipeline};
use anyhow::Result;
use ash::vk;

impl Context {
    pub fn submit_and_wait(&self, record: impl FnOnce(vk::CommandBuffer)) -> Result<()> {
        let device = &self.device;
        let alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe { device.allocate_command_buffers(&alloc) }?[0];
        let begin = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            device.begin_command_buffer(cmd, &begin)?;
            record(cmd);
            device.end_command_buffer(cmd)?;
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let cmds = [cmd];
            let submit = [vk::SubmitInfo::default().command_buffers(&cmds)];
            let result = device
                .queue_submit(self.queue, &submit, fence)
                .and_then(|_| device.wait_for_fences(&[fence], true, u64::MAX));
            device.destroy_fence(fence, None);
            device.free_command_buffers(self.cmd_pool, &cmds);
            result?;
        }
        Ok(())
    }

    pub fn time_dispatch(&self, pipeline: &Pipeline, groups: [u32; 3], push: &[u8], reps: u32) -> Result<Vec<f64>> {
        let device = &self.device;
        let pool_info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(reps * 2);
        let pool = unsafe { device.create_query_pool(&pool_info, None) }?;
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
        let compute = vk::PipelineStageFlags::COMPUTE_SHADER;

        let result = self.submit_and_wait(|cmd| unsafe {
            device.cmd_reset_query_pool(cmd, pool, 0, reps * 2);
            pipeline.bind(cmd);
            if !push.is_empty() {
                device.cmd_push_constants(cmd, pipeline.layout, vk::ShaderStageFlags::COMPUTE, 0, push);
            }
            for r in 0..reps {
                device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, pool, r * 2);
                device.cmd_dispatch(cmd, groups[0], groups[1], groups[2]);
                device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, pool, r * 2 + 1);
                device.cmd_pipeline_barrier(
                    cmd,
                    compute,
                    compute,
                    vk::DependencyFlags::empty(),
                    &[barrier],
                    &[],
                    &[],
                );
            }
        });

        let mut stamps = vec![0u64; (reps * 2) as usize];
        let flags = vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT;
        let read = result
            .and_then(|_| unsafe { device.get_query_pool_results(pool, 0, &mut stamps, flags) }.map_err(Into::into));
        unsafe { device.destroy_query_pool(pool, None) };
        read?;

        let period = self.props.limits.timestamp_period as f64;
        Ok(stamps
            .as_chunks::<2>()
            .0
            .iter()
            .map(|[start, end]| end.wrapping_sub(*start) as f64 * period)
            .collect())
    }
}
