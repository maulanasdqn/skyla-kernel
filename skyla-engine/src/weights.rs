use crate::host_memory;
use crate::planner::Planner;
use anyhow::{anyhow, Result};
use skyla_gguf::{DType, Gguf};
use skyla_ref::Config;
use skyla_vk::{Buffer, BufferUsageFlags, Context};

#[derive(Clone, Copy)]
pub struct QMatrix {
    pub dtype: DType,
    pub buf: usize,
    pub off: [u32; 4],
    pub n_in: u32,
    pub n_out: u32,
}

pub struct LayerWeights {
    pub attn_norm: u32,
    pub q_norm: u32,
    pub k_norm: u32,
    pub ffn_norm: u32,
    pub wq: QMatrix,
    pub wk: QMatrix,
    pub wv: QMatrix,
    pub wo: QMatrix,
    pub gate: QMatrix,
    pub up: QMatrix,
    pub down: QMatrix,
}

pub struct Weights {
    pub buffers: Vec<Buffer>,
    pub output: QMatrix,
    pub out_norm: u32,
    pub rope: u32,
    pub layers: Vec<LayerWeights>,
}

impl Weights {
    pub fn load(ctx: &Context, g: &Gguf, c: &Config) -> Result<Self> {
        let cap = ctx
            .max_allocation_size
            .min(ctx.props.limits.max_storage_buffer_range as u64);
        let mut p = Planner::new(g, cap as usize);
        let rope = p.table(c.rope_freqs.clone());
        let out_norm = p.norm("output_norm.weight")?;
        let mut norms = Vec::with_capacity(c.n_layer);
        for i in 0..c.n_layer {
            let n = |s: &str| format!("blk.{i}.{s}.weight");
            norms.push([
                p.norm(&n("attn_norm"))?,
                p.norm(&n("attn_q_norm"))?,
                p.norm(&n("attn_k_norm"))?,
                p.norm(&n("ffn_norm"))?,
            ]);
        }
        let output = match g.tensors.contains_key("output.weight") {
            true => p.matrix("output.weight")?,
            false => p.matrix("token_embd.weight")?,
        };
        let mut layers = Vec::with_capacity(c.n_layer);
        for (i, [attn_norm, q_norm, k_norm, ffn_norm]) in norms.into_iter().enumerate() {
            let n = |s: &str| format!("blk.{i}.{s}.weight");
            layers.push(LayerWeights {
                attn_norm,
                q_norm,
                k_norm,
                ffn_norm,
                wq: p.matrix(&n("attn_q"))?,
                wk: p.matrix(&n("attn_k"))?,
                wv: p.matrix(&n("attn_v"))?,
                wo: p.matrix(&n("attn_output"))?,
                gate: p.matrix(&n("ffn_gate"))?,
                up: p.matrix(&n("ffn_up"))?,
                down: p.matrix(&n("ffn_down"))?,
            });
        }

        let memory = host_memory(ctx)?;
        let mut buffers = Vec::with_capacity(p.sizes.len());
        for (i, &size) in p.sizes.iter().enumerate() {
            let size = size.next_multiple_of(16) as u64;
            let mut b = ctx.alloc_buffer(size, BufferUsageFlags::STORAGE_BUFFER, memory)?;
            p.fill(
                i,
                b.as_mut_slice().ok_or_else(|| anyhow!("weight buffer is not mapped"))?,
            );
            buffers.push(b);
        }
        Ok(Self {
            buffers,
            output,
            out_norm,
            rope,
            layers,
        })
    }
}
