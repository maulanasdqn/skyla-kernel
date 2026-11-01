use crate::arena::ATTENTION_CHUNK;
use anyhow::Result;
use skyla_vk::{Buffer, Context, Pipeline};

pub const WAVES_PER_GROUP: u32 = 4;
pub const ROWS_PER_WAVE: u32 = 2;
pub const ROWS_PER_GROUP: u32 = WAVES_PER_GROUP * ROWS_PER_WAVE;

pub struct Kernels {
    pub matvec: Pipeline,
    pub matvec_acc: Pipeline,
    pub rmsnorm: Pipeline,
    pub qkv: Pipeline,
    pub attention: Pipeline,
    pub merge: Pipeline,
    pub silu: Pipeline,
    pub embed: Pipeline,
}

impl Kernels {
    pub fn new(ctx: &Context, w: &Buffer, a: &Buffer, kv: &Buffer, n_rep: u32) -> Result<Self> {
        let build = |spirv: &[u8], bufs: &[&Buffer], push: u32, spec: &[u32]| -> Result<Pipeline> {
            let p = ctx.create_pipeline(spirv, bufs.len() as u32, push, spec)?;
            p.bind_buffers(bufs);
            Ok(p)
        };
        use skyla_shaders as s;
        Ok(Self {
            matvec: build(s::MATVEC_Q8_0, &[w, w, a, a], 24, &[WAVES_PER_GROUP, 0, ROWS_PER_WAVE])?,
            matvec_acc: build(s::MATVEC_Q8_0, &[w, w, a, a], 24, &[WAVES_PER_GROUP, 1, ROWS_PER_WAVE])?,
            rmsnorm: build(s::RMSNORM, &[w, a], 20, &[])?,
            qkv: build(s::QKV_PREP, &[w, a, a, kv], 44, &[])?,
            attention: build(s::ATTENTION, &[a, a, kv], 28, &[n_rep, ATTENTION_CHUNK])?,
            merge: build(s::ATTENTION_MERGE, &[a, a], 16, &[])?,
            silu: build(s::SILU_MUL, &[a], 12, &[])?,
            embed: build(s::EMBED_Q8_0, &[w, w, a, a], 16, &[])?,
        })
    }
}
