use crate::{Context, Pipeline};
use anyhow::{ensure, Result};
use ash::vk;
use std::cell::RefCell;

const MAX_MARKS: u32 = 2048;

pub struct Recording {
    device: ash::Device,
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    queries: vk::QueryPool,
    labels: Vec<&'static str>,
}

pub struct Recorder<'a> {
    device: &'a ash::Device,
    cmd: vk::CommandBuffer,
    queries: vk::QueryPool,
    labels: RefCell<Vec<&'static str>>,
}

impl Recorder<'_> {
    pub fn dispatch(&self, p: &Pipeline, groups: [u32; 3], push: &[u32]) {
        p.bind(self.cmd);
        if !push.is_empty() {
            let bytes: Vec<u8> = push.iter().flat_map(|w| w.to_le_bytes()).collect();
            unsafe {
                self.device
                    .cmd_push_constants(self.cmd, p.layout, vk::ShaderStageFlags::COMPUTE, 0, &bytes)
            };
        }
        unsafe { self.device.cmd_dispatch(self.cmd, groups[0], groups[1], groups[2]) };
    }

    pub fn barrier(&self) {
        self.memory_barrier(
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
        );
    }

    pub fn host_barrier(&self) {
        self.memory_barrier(vk::PipelineStageFlags::HOST, vk::AccessFlags::HOST_READ);
    }

    pub fn mark(&self, label: &'static str) {
        if self.queries == vk::QueryPool::null() {
            return;
        }
        let mut labels = self.labels.borrow_mut();
        let index = labels.len() as u32 + 1;
        assert!(index < MAX_MARKS, "too many profiling marks");
        let stage = vk::PipelineStageFlags::BOTTOM_OF_PIPE;
        unsafe { self.device.cmd_write_timestamp(self.cmd, stage, self.queries, index) };
        labels.push(label);
    }

    fn memory_barrier(&self, dst_stage: vk::PipelineStageFlags, dst_access: vk::AccessFlags) {
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(dst_access);
        unsafe {
            self.device.cmd_pipeline_barrier(
                self.cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            )
        };
    }
}

impl Context {
    pub fn record(&self, profile: bool, f: impl FnOnce(&Recorder)) -> Result<Recording> {
        let device = &self.device;
        let alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe { device.allocate_command_buffers(&alloc) }?[0];
        let queries = match profile {
            true => {
                let info = vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(MAX_MARKS);
                unsafe { device.create_query_pool(&info, None) }?
            }
            false => vk::QueryPool::null(),
        };

        unsafe { device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()) }?;
        if profile {
            unsafe {
                device.cmd_reset_query_pool(cmd, queries, 0, MAX_MARKS);
                device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, queries, 0);
            }
        }
        let recorder = Recorder {
            device,
            cmd,
            queries,
            labels: RefCell::new(Vec::new()),
        };
        f(&recorder);
        unsafe { device.end_command_buffer(cmd) }?;
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }?;
        Ok(Recording {
            device: device.clone(),
            pool: self.cmd_pool,
            cmd,
            fence,
            queries,
            labels: recorder.labels.into_inner(),
        })
    }

    pub fn replay(&self, r: &Recording) -> Result<()> {
        let cmds = [r.cmd];
        let submit = [vk::SubmitInfo::default().command_buffers(&cmds)];
        unsafe {
            self.device.reset_fences(&[r.fence])?;
            self.device.queue_submit(self.queue, &submit, r.fence)?;
            self.device.wait_for_fences(&[r.fence], true, u64::MAX)?;
        }
        Ok(())
    }

    pub fn marks(&self, r: &Recording) -> Result<Vec<(&'static str, f64)>> {
        ensure!(r.queries != vk::QueryPool::null(), "recording was not profiled");
        let mut stamps = vec![0u64; r.labels.len() + 1];
        let flags = vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT;
        unsafe { self.device.get_query_pool_results(r.queries, 0, &mut stamps, flags) }?;
        let period = self.props.limits.timestamp_period as f64;
        Ok(r.labels
            .iter()
            .zip(stamps.windows(2))
            .map(|(&label, w)| (label, w[1].wrapping_sub(w[0]) as f64 * period))
            .collect())
    }
}

impl Drop for Recording {
    fn drop(&mut self) {
        unsafe {
            if self.queries != vk::QueryPool::null() {
                self.device.destroy_query_pool(self.queries, None);
            }
            self.device.destroy_fence(self.fence, None);
            self.device.free_command_buffers(self.pool, &[self.cmd]);
        }
    }
}
