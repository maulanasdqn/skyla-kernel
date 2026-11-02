use crate::buffer::Raw;
use crate::context::State;
use crate::{Buffer, Context};
use anyhow::Result;
use skyla_shaders::Shader;
use std::cell::RefCell;

pub struct Pipeline {
    pub(crate) state: State,
    pub(crate) threads: [u32; 3],
    sets: RefCell<Vec<Vec<Raw>>>,
}

fn threads(name: &str, spec: &[u32]) -> [u32; 3] {
    let first = spec.first().copied().unwrap_or(1);
    match name {
        "matvec_q4k" | "matvec_q6k" | "matvec_q8_0" => [32, first, 1],
        "attention" | "qkv_prep" | "moe_logits" => [32, 1, 1],
        "attention_merge" => [128, 1, 1],
        "rmsnorm" => [1024, 1, 1],
        "bandwidth" => [first, 1, 1],
        _ => [256, 1, 1],
    }
}

impl Context {
    pub fn create_pipeline(&self, shader: &Shader, n_buffers: u32, push_size: u32, spec: &[u32]) -> Result<Pipeline> {
        self.create_pipeline_sets(shader, n_buffers, push_size, spec, 1)
    }

    pub fn create_pipeline_sets(
        &self,
        shader: &Shader,
        _n_buffers: u32,
        _push_size: u32,
        spec: &[u32],
        n_sets: u32,
    ) -> Result<Pipeline> {
        Ok(Pipeline {
            state: self.state(shader.name, spec)?,
            threads: threads(shader.name, spec),
            sets: RefCell::new(vec![Vec::new(); n_sets as usize]),
        })
    }
}

impl Pipeline {
    pub fn bind_buffers(&self, buffers: &[&Buffer]) {
        self.bind_set_buffers(0, buffers);
    }

    pub fn bind_set_buffers(&self, set: usize, buffers: &[&Buffer]) {
        self.sets.borrow_mut()[set] = buffers.iter().map(|b| b.raw.clone()).collect();
    }

    pub(crate) fn set(&self, set: usize) -> Vec<Raw> {
        self.sets.borrow()[set].clone()
    }
}
