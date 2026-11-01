use crate::arena::ATTENTION_CHUNK;
use crate::kv::KvPrecision;
use anyhow::{bail, Result};
use skyla_gguf::DType;
use skyla_vk::{Buffer, Context, Pipeline};

pub const Q8_0_TACTIC: (u32, u32) = (4, 2);
pub const KQUANT_TACTIC: (u32, u32) = (2, 4);
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
    tactic: (u32, u32),
}

fn matvec(ctx: &Context, kind: Kind, weights: &[Buffer], a: &Buffer) -> Result<Matvec> {
    let (waves, rows) = kind.tactic;
    let build = |accum: u32| -> Result<Pipeline> {
        let spec = match kind.matmul_type {
            Some(ty) => vec![ty, accum],
            None => vec![waves, accum, rows],
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
            None => waves * rows,
        },
    })
}

fn wave32(ctx: &Context) -> Result<bool> {
    match ctx.subgroup_size {
        64 => Ok(false),
        32 => Ok(true),
        n => bail!("no kernels for subgroup size {n} (need 32 or 64)"),
    }
}

impl Kernels {
    pub fn new(
        ctx: &Context,
        weights: &[Buffer],
        a: &Buffer,
        kv: &Buffer,
        n_rep: u32,
        precision: KvPrecision,
    ) -> Result<Self> {
        let (k8, v8) = (precision.keys_q8() as u32, precision.values_q8() as u32);
        let w32 = wave32(ctx)?;
        let pick = |w64: &'static [u8], w32_spirv: &'static [u8]| if w32 { w32_spirv } else { w64 };
        let build = |spirv: &[u8], bufs: &[&Buffer], push: u32, spec: &[u32]| -> Result<Pipeline> {
            let p = ctx.create_pipeline(spirv, bufs.len() as u32, push, spec)?;
            p.bind_buffers(bufs);
            Ok(p)
        };
        use skyla_shaders as s;
        let kind = |spirv, dual_view, push, matmul_type, tactic| Kind {
            spirv,
            dual_view,
            push,
            matmul_type,
            tactic,
        };
        let decode = |w64, w32, dual_view, push, tactic| {
            matvec(ctx, kind(pick(w64, w32), dual_view, push, None, tactic), weights, a)
        };
        let w0 = &weights[0];
        let mm = (0, 0);
        Ok(Self {
            decode: [
                decode(s::MATVEC_Q8_0, s::MATVEC_Q8_0_W32, true, 24, Q8_0_TACTIC)?,
                decode(s::MATVEC_Q4K, s::MATVEC_Q4K_W32, false, 24, KQUANT_TACTIC)?,
                decode(s::MATVEC_Q6K, s::MATVEC_Q6K_W32, true, 32, KQUANT_TACTIC)?,
            ],
            batch: [
                matvec(ctx, kind(s::MATMUL, true, 40, Some(0), mm), weights, a)?,
                matvec(ctx, kind(s::MATMUL, true, 40, Some(1), mm), weights, a)?,
                matvec(ctx, kind(s::MATMUL, true, 40, Some(2), mm), weights, a)?,
            ],
            rmsnorm: build(pick(s::RMSNORM, s::RMSNORM_W32), &[w0, a, a], 24, &[])?,
            qkv: build(pick(s::QKV_PREP, s::QKV_PREP_W32), &[w0, a, a, kv, kv], 64, &[k8, v8])?,
            attention: build(
                pick(s::ATTENTION, s::ATTENTION_W32),
                &[a, a, kv, kv, kv],
                44,
                &[n_rep, ATTENTION_CHUNK, k8, v8],
            )?,
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
