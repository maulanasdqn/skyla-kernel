use crate::gpu::{Buffer, Context};
use crate::planner::Planner;
use anyhow::{anyhow, Result};
use skyla_gguf::{DType, Gguf};
use skyla_ref::Config;

#[derive(Clone, Copy)]
pub struct QMatrix {
    pub dtype: DType,
    pub buf: usize,
    pub off: [u32; 4],
    pub n_in: u32,
    pub n_out: u32,
}

pub struct Attention {
    pub q_norm: u32,
    pub k_norm: u32,
    pub wq: QMatrix,
    pub wk: QMatrix,
    pub wv: QMatrix,
    pub wo: QMatrix,
}

pub struct DeltaNet {
    pub conv: u32,
    pub a: u32,
    pub dt: u32,
    pub norm: u32,
    pub qkv: QMatrix,
    pub z: QMatrix,
    pub alpha: QMatrix,
    pub beta: QMatrix,
    pub out: QMatrix,
}

pub enum Mixer {
    Attention(Attention),
    DeltaNet(DeltaNet),
}

pub struct LayerWeights {
    pub attn_norm: u32,
    pub ffn_norm: u32,
    pub mixer: Mixer,
    pub gate: QMatrix,
    pub up: QMatrix,
    pub down: QMatrix,
    pub router: Option<u32>,
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
        let cap = ctx.max_buffer();
        let mut p = Planner::new(g, cap as usize);
        let rope = p.table(c.rope_freqs.clone());
        let out_norm = p.norm("output_norm.weight")?;
        let moe = c.n_expert > 0;
        let mut tables = Vec::with_capacity(c.n_layer);
        let mut routers = Vec::with_capacity(c.n_layer);
        for i in 0..c.n_layer {
            let n = |s: &str| format!("blk.{i}.{s}.weight");
            let norms = [p.norm(&n("attn_norm"))?, p.norm(&n(c.ffn_norm()))?];
            let mixer = match c.recurrent(i) {
                true => [
                    p.norm(&n("ssm_conv1d"))?,
                    p.norm(&format!("blk.{i}.ssm_a"))?,
                    p.norm(&format!("blk.{i}.ssm_dt.bias"))?,
                    p.norm(&n("ssm_norm"))?,
                ],
                false => [p.norm(&n("attn_q_norm"))?, p.norm(&n("attn_k_norm"))?, 0, 0],
            };
            tables.push((norms, mixer));
            routers.push(match moe {
                true => Some(p.norm(&n("ffn_gate_inp"))?),
                false => None,
            });
        }
        let ffn = |s: &str| if moe { format!("{s}_exps") } else { s.to_string() };
        let output = match g.tensors.contains_key("output.weight") {
            true => p.matrix("output.weight")?,
            false => p.matrix("token_embd.weight")?,
        };
        let mut layers = Vec::with_capacity(c.n_layer);
        for (i, ([attn_norm, ffn_norm], [t0, t1, t2, t3])) in tables.into_iter().enumerate() {
            let n = |s: &str| format!("blk.{i}.{s}.weight");
            let mixer = match c.recurrent(i) {
                true => Mixer::DeltaNet(DeltaNet {
                    conv: t0,
                    a: t1,
                    dt: t2,
                    norm: t3,
                    qkv: p.matrix(&n("attn_qkv"))?,
                    z: p.matrix(&n("attn_gate"))?,
                    alpha: p.matrix(&n("ssm_alpha"))?,
                    beta: p.matrix(&n("ssm_beta"))?,
                    out: p.matrix(&n("ssm_out"))?,
                }),
                false => Mixer::Attention(Attention {
                    q_norm: t0,
                    k_norm: t1,
                    wq: p.matrix(&n("attn_q"))?,
                    wk: p.matrix(&n("attn_k"))?,
                    wv: p.matrix(&n("attn_v"))?,
                    wo: p.matrix(&n("attn_output"))?,
                }),
            };
            layers.push(LayerWeights {
                attn_norm,
                ffn_norm,
                mixer,
                gate: p.matrix(&n(&ffn("ffn_gate")))?,
                up: p.matrix(&n(&ffn("ffn_up")))?,
                down: p.matrix(&n(&ffn("ffn_down")))?,
                router: routers[i],
            });
        }

        let mut buffers = Vec::with_capacity(p.sizes.len());
        for (i, &size) in p.sizes.iter().enumerate() {
            let size = size.next_multiple_of(16) as u64;
            let mut b = ctx.alloc_host(size)?;
            p.fill(
                i,
                b.as_mut_slice().ok_or_else(|| anyhow!("weight buffer is not mapped"))?,
            )?;
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
