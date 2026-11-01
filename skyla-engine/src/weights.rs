use crate::host_memory;
use anyhow::{anyhow, ensure, Result};
use skyla_gguf::{DType, Gguf};
use skyla_vk::{Buffer, BufferUsageFlags, Context};

#[derive(Clone, Copy)]
pub struct QMatrix {
    pub q: u32,
    pub d: u32,
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
    pub buffer: Buffer,
    pub embd: QMatrix,
    pub output: QMatrix,
    pub out_norm: u32,
    pub layers: Vec<LayerWeights>,
}

struct Planner<'a> {
    g: &'a Gguf,
    size: usize,
    norms: Vec<(usize, String)>,
    mats: Vec<(usize, usize, &'a [u8])>,
}

impl<'a> Planner<'a> {
    fn alloc(&mut self, bytes: usize) -> usize {
        let off = self.size.next_multiple_of(16);
        self.size = off + bytes;
        off
    }

    fn norm(&mut self, name: &str) -> Result<u32> {
        let n = self.g.tensor(name)?.n_elements() as usize;
        let off = self.alloc(n * 4);
        self.norms.push((off, name.to_string()));
        Ok((off / 4) as u32)
    }

    fn matrix(&mut self, name: &str) -> Result<QMatrix> {
        let t = self.g.tensor(name)?;
        ensure!(
            t.dtype == DType::Q8_0,
            "{name} is {:?}; only Q8_0 is packed so far",
            t.dtype
        );
        let (n_in, n_out) = (t.dims[0] as usize, t.dims[1] as usize);
        ensure!(n_in % 16 == 0, "{name}: input size {n_in} is not a multiple of 16");
        let blocks = n_in * n_out / 32;
        let q = self.alloc(blocks * 32);
        let d = self.alloc(blocks * 2);
        self.mats.push((q, d, self.g.data(t)));
        Ok(QMatrix {
            q: (q / 16) as u32,
            d: (d / 2) as u32,
            n_in: n_in as u32,
            n_out: n_out as u32,
        })
    }

    fn fill(&self, dst: &mut [u8]) -> Result<()> {
        for (off, name) in &self.norms {
            let v = self.g.to_f32(name)?;
            for (i, x) in v.iter().enumerate() {
                dst[off + i * 4..off + i * 4 + 4].copy_from_slice(&x.to_le_bytes());
            }
        }
        for &(q, d, src) in &self.mats {
            for (i, b) in src.as_chunks::<34>().0.iter().enumerate() {
                dst[d + i * 2..d + i * 2 + 2].copy_from_slice(&b[..2]);
                dst[q + i * 32..q + i * 32 + 32].copy_from_slice(&b[2..]);
            }
        }
        Ok(())
    }
}

impl Weights {
    pub fn load(ctx: &Context, g: &Gguf, n_layer: usize) -> Result<Self> {
        let mut p = Planner {
            g,
            size: 0,
            norms: Vec::new(),
            mats: Vec::new(),
        };
        let embd = p.matrix("token_embd.weight")?;
        let output = match g.tensors.contains_key("output.weight") {
            true => p.matrix("output.weight")?,
            false => embd,
        };
        let out_norm = p.norm("output_norm.weight")?;
        let layers = (0..n_layer)
            .map(|i| {
                let n = |s: &str| format!("blk.{i}.{s}.weight");
                Ok(LayerWeights {
                    attn_norm: p.norm(&n("attn_norm"))?,
                    q_norm: p.norm(&n("attn_q_norm"))?,
                    k_norm: p.norm(&n("attn_k_norm"))?,
                    ffn_norm: p.norm(&n("ffn_norm"))?,
                    wq: p.matrix(&n("attn_q"))?,
                    wk: p.matrix(&n("attn_k"))?,
                    wv: p.matrix(&n("attn_v"))?,
                    wo: p.matrix(&n("attn_output"))?,
                    gate: p.matrix(&n("ffn_gate"))?,
                    up: p.matrix(&n("ffn_up"))?,
                    down: p.matrix(&n("ffn_down"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let size = p.size.next_multiple_of(16) as u64;
        ensure!(
            size <= ctx.max_allocation_size,
            "weights need {size} bytes, above the allocation cap"
        );
        let mut buffer = ctx.alloc_buffer(size, BufferUsageFlags::STORAGE_BUFFER, host_memory(ctx)?)?;
        let dst = buffer
            .as_mut_slice()
            .ok_or_else(|| anyhow!("weight buffer is not mapped"))?;
        p.fill(dst)?;
        Ok(Self {
            buffer,
            embd,
            output,
            out_norm,
            layers,
        })
    }
}
