use crate::arena::ATTENTION_CHUNK;
use anyhow::Result;
use skyla_gguf::DType;
use skyla_vk::{Buffer, Context, Pipeline};

pub const WAVES_PER_GROUP: u32 = 4;
pub const ROWS_PER_WAVE: u32 = 2;
pub const MATMUL_TILE_ROWS: u32 = 128;

pub struct Matvec {
    plain: Pipeline,
    acc: Pipeline,
    pub rows_per_group: u32,
}

pub struct Kernels {
    decode: [Matvec; 3],
    batch: [Matvec; 3],
    pub rmsnorm: Pipeline,
    pub qkv: Pipeline,
    pub attention: Pipeline,
    pub merge: Pipeline,
    pub silu: Pipeline,
    pub take_last: Pipeline,
}

struct Kind<'a> {
    spirv: &'a [u8],
    dual_view: bool,
    push: u32,
    matmul_type: Option<u32>,
}

fn matvec(ctx: &Context, kind: Kind, weights: &[Buffer], a: &Buffer) -> Result<Matvec> {
    let build = |accum: u32| -> Result<Pipeline> {
        let spec = match kind.matmul_type {
            Some(ty) => vec![ty, accum],
            None => vec![WAVES_PER_GROUP, accum, ROWS_PER_WAVE],
        };
        let batched = kind.matmul_type.is_some();
        let n_buffers = 2 + kind.dual_view as u32 + 1 + batched as u32;
        let p = ctx.create_pipeline_sets(kind.spirv, n_buffers, kind.push, &spec, weights.len() as u32)?;
        for (i, w) in weights.iter().enumerate() {
            let mut bufs = vec![w];
            if kind.dual_view {
                bufs.push(w);
            }
            bufs.extend([a, a]);
            if batched {
                bufs.push(a);
            }
            p.bind_set_buffers(i, &bufs);
        }
        Ok(p)
    };
    Ok(Matvec {
        plain: build(0)?,
        acc: build(1)?,
        rows_per_group: match kind.matmul_type {
            Some(_) => MATMUL_TILE_ROWS,
            None => WAVES_PER_GROUP * ROWS_PER_WAVE,
        },
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
        let kind = |spirv, dual_view, push, matmul_type| Kind {
            spirv,
            dual_view,
            push,
            matmul_type,
        };
        let w0 = &weights[0];
        Ok(Self {
            decode: [
                matvec(ctx, kind(s::MATVEC_Q8_0, true, 24, None), weights, a)?,
                matvec(ctx, kind(s::MATVEC_Q4K, false, 24, None), weights, a)?,
                matvec(ctx, kind(s::MATVEC_Q6K, true, 32, None), weights, a)?,
            ],
            batch: [
                matvec(ctx, kind(s::MATMUL, true, 40, Some(0)), weights, a)?,
                matvec(ctx, kind(s::MATMUL, true, 40, Some(1)), weights, a)?,
                matvec(ctx, kind(s::MATMUL, true, 40, Some(2)), weights, a)?,
            ],
            rmsnorm: build(s::RMSNORM, &[w0, a, a], 24, &[])?,
            qkv: build(s::QKV_PREP, &[w0, a, a, kv], 56, &[])?,
            attention: build(s::ATTENTION, &[a, a, kv], 36, &[n_rep, ATTENTION_CHUNK])?,
            merge: build(s::ATTENTION_MERGE, &[a, a], 24, &[])?,
            silu: build(s::SILU_MUL, &[a, a], 16, &[])?,
            take_last: build(s::TAKE_LAST, &[a, a], 16, &[])?,
        })
    }

    pub fn matvec(&self, dtype: DType, batched: bool) -> &Matvec {
        let set = if batched { &self.batch } else { &self.decode };
        match dtype {
            DType::Q4K => &set[1],
            DType::Q6K => &set[2],
            _ => &set[0],
        }
    }
}

impl Matvec {
    pub fn pipeline(&self, accum: bool) -> &Pipeline {
        if accum {
            &self.acc
        } else {
            &self.plain
        }
    }
}
