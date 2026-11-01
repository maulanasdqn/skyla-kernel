use crate::arena::Arena;
use crate::graph;
use crate::kernels::Kernels;
use crate::weights::Weights;
use anyhow::{anyhow, ensure, Result};
use skyla_gguf::{dequantize, DType, Gguf};
use skyla_ref::Config;
use skyla_vk::{Context, Recording};

pub struct Engine<'g> {
    recording: Recording,
    _kernels: Kernels,
    arena: Arena,
    weights: Weights,
    embd: (DType, &'g [u8], usize),
    pub config: Config,
    pub pos: u32,
    ctx: Context,
}

impl<'g> Engine<'g> {
    pub fn new(g: &'g Gguf, max_ctx: usize, profile: bool) -> Result<Self> {
        let config = Config::from_gguf(g)?;
        ensure!(
            config.head_dim == 128,
            "kernels assume head_dim 128, got {}",
            config.head_dim
        );
        ensure!(
            matches!(config.n_head / config.n_head_kv, 1..=8),
            "unsupported GQA ratio"
        );
        let t = g.tensor("token_embd.weight")?;
        let embd = (t.dtype, g.data(t), t.dtype.row_bytes(config.n_embd)?);

        let ctx = Context::new()?;
        let weights = Weights::load(&ctx, g, &config)?;
        let arena = Arena::new(&ctx, &config, max_ctx)?;
        let n_rep = (config.n_head / config.n_head_kv) as u32;
        let kernels = Kernels::new(&ctx, &weights.buffers, &arena.buffer, &arena.kv, n_rep)?;
        let recording = graph::record(&ctx, &kernels, &weights, &arena, &config, profile)?;
        Ok(Self {
            recording,
            _kernels: kernels,
            arena,
            weights,
            embd,
            config,
            pos: 0,
            ctx,
        })
    }

    pub fn device_name(&self) -> String {
        self.ctx.device_name()
    }

    pub fn weight_bytes(&self) -> u64 {
        self.weights.buffers.iter().map(|b| b.size).sum()
    }

    pub fn weight_buffers(&self) -> usize {
        self.weights.buffers.len()
    }

    pub fn reset(&mut self) {
        self.pos = 0;
    }

    pub fn last_step_profile(&self) -> Result<Vec<(&'static str, f64)>> {
        self.ctx.marks(&self.recording)
    }

    pub fn step(&mut self, token: u32) -> Result<&[f32]> {
        let max = self.arena.max_ctx;
        ensure!(self.pos < max, "context full ({max} tokens)");
        let (dtype, data, row) = self.embd;
        let x = self.arena.x as usize;
        let arena = self
            .arena
            .buffer
            .as_f32_mut()
            .ok_or_else(|| anyhow!("arena is not mapped"))?;
        let src = data
            .get(token as usize * row..(token as usize + 1) * row)
            .ok_or_else(|| anyhow!("token {token} is outside the vocabulary"))?;
        dequantize(dtype, src, &mut arena[x..x + self.config.n_embd])?;
        let ctl = self
            .arena
            .buffer
            .as_u32_mut()
            .ok_or_else(|| anyhow!("arena is not mapped"))?;
        ctl[0] = token;
        ctl[1] = self.pos;

        self.ctx.replay(&self.recording)?;
        self.pos += 1;
        let all = self
            .arena
            .buffer
            .as_f32()
            .ok_or_else(|| anyhow!("arena is not mapped"))?;
        let start = self.arena.logits as usize;
        Ok(&all[start..start + self.config.n_vocab])
    }
}
