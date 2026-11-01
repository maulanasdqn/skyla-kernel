use crate::arena::ATTENTION_CHUNK;
use anyhow::Result;
use skyla_gguf::DType;
use skyla_vk::{Buffer, Context, Pipeline};

pub const WAVES_PER_GROUP: u32 = 4;
pub const ROWS_PER_WAVE: u32 = 2;
pub const ROWS_PER_GROUP: u32 = WAVES_PER_GROUP * ROWS_PER_WAVE;

pub struct Matvec {
    plain: Pipeline,
    acc: Pipeline,
}

pub struct Kernels {
    q8_0: Matvec,
    q4k: Matvec,
    q6k: Matvec,
    pub rmsnorm: Pipeline,
    pub qkv: Pipeline,
    pub attention: Pipeline,
    pub merge: Pipeline,
    pub silu: Pipeline,
}

fn matvec(ctx: &Context, spirv: &[u8], weights: &[Buffer], a: &Buffer, dual_view: bool, push: u32) -> Result<Matvec> {
    let n_sets = weights.len() as u32;
    let build = |accum: u32| -> Result<Pipeline> {
        let n_buffers = if dual_view { 4 } else { 3 };
        let spec = [WAVES_PER_GROUP, accum, ROWS_PER_WAVE];
        let p = ctx.create_pipeline_sets(spirv, n_buffers, push, &spec, n_sets)?;
        for (i, w) in weights.iter().enumerate() {
            match dual_view {
                true => p.bind_set_buffers(i, &[w, w, a, a]),
                false => p.bind_set_buffers(i, &[w, a, a]),
            }
        }
        Ok(p)
    };
    Ok(Matvec {
        plain: build(0)?,
        acc: build(1)?,
    })
}

impl Kernels {
    pub fn new(ctx: &Context, weights: &[Buffer], a: &Buffer, kv: &Buffer, n_rep: u32) -> Result<Self> {
        let build = |spirv: &[u8], bufs: &[&Buffer], push: u32, spec: &[u32]| -> Result<Pipeline> {
            let p = ctx.create_pipeline(spirv, bufs.len() as u32, push, spec)?;
            p.bind_buffers(bufs);
            Ok(p)
        };
        use skyla_shaders as s;
        let w0 = &weights[0];
        Ok(Self {
            q8_0: matvec(ctx, s::MATVEC_Q8_0, weights, a, true, 24)?,
            q4k: matvec(ctx, s::MATVEC_Q4K, weights, a, false, 24)?,
            q6k: matvec(ctx, s::MATVEC_Q6K, weights, a, true, 32)?,
            rmsnorm: build(s::RMSNORM, &[w0, a], 20, &[])?,
            qkv: build(s::QKV_PREP, &[w0, a, a, kv], 48, &[])?,
            attention: build(s::ATTENTION, &[a, a, kv], 28, &[n_rep, ATTENTION_CHUNK])?,
            merge: build(s::ATTENTION_MERGE, &[a, a], 16, &[])?,
            silu: build(s::SILU_MUL, &[a], 12, &[])?,
        })
    }

    pub fn matvec(&self, dtype: DType, accum: bool) -> &Pipeline {
        let m = match dtype {
            DType::Q4K => &self.q4k,
            DType::Q6K => &self.q6k,
            _ => &self.q8_0,
        };
        if accum {
            &m.acc
        } else {
            &m.plain
        }
    }
}
