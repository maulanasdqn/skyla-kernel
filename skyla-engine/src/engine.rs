use crate::arena::{Arena, BATCH};
use crate::graph;
use crate::kernels::Kernels;
use crate::kv::KvPrecision;
use crate::weights::Weights;
use anyhow::{anyhow, ensure, Result};
use skyla_gguf::{dequantize, DType, Gguf};
use skyla_ref::Config;
use skyla_vk::{Context, Recording};

pub struct Engine<'g> {
    decode: Recording,
    prefill: Recording,
    _kernels: Kernels,
    arena: Arena,
    weights: Weights,
    embd: (DType, &'g [u8], usize),
    pub config: Config,
    pub pos: u32,
    ctx: Context,
}

impl<'g> Engine<'g> {
    pub fn new(g: &'g Gguf, max_ctx: usize, kv: KvPrecision, profile: bool) -> Result<Self> {
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
        let arena = Arena::new(&ctx, &config, max_ctx, kv)?;
        let n_rep = (config.n_head / config.n_head_kv) as u32;
        let kernels = Kernels::new(&ctx, &weights.buffers, &arena.buffer, &arena.kv, n_rep, kv)?;
        let decode = graph::record(&ctx, &kernels, &weights, &arena, &config, false, profile)?;
        let prefill = graph::record(&ctx, &kernels, &weights, &arena, &config, true, profile)?;

        Ok(Self {
            decode,
            prefill,
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

    pub fn last_step_profile(&self, prefill: bool) -> Result<Vec<(&'static str, f64)>> {
        match prefill {
            true => self.ctx.marks(&self.prefill),
            false => self.ctx.marks(&self.decode),
        }
    }

    fn load(&mut self, tokens: &[u32]) -> Result<()> {
        let max = self.arena.max_ctx;
        ensure!(self.pos + tokens.len() as u32 <= max, "context full ({max} tokens)");
        let (dtype, data, row) = self.embd;
        let n = self.config.n_embd;
        let (x, stride) = (self.arena.x as usize, self.arena.s_embd as usize);
        let arena = self
            .arena
            .buffer
            .as_f32_mut()
            .ok_or_else(|| anyhow!("arena is not mapped"))?;
        for (t, &token) in tokens.iter().enumerate() {
            let src = data
                .get(token as usize * row..(token as usize + 1) * row)
                .ok_or_else(|| anyhow!("token {token} is outside the vocabulary"))?;
            let dst = x + t * stride;
            dequantize(dtype, src, &mut arena[dst..dst + n])?;
        }
        let ctl = self
            .arena
            .buffer
            .as_u32_mut()
            .ok_or_else(|| anyhow!("arena is not mapped"))?;
        ctl[0] = tokens[0];
        ctl[1] = self.pos;
        ctl[2] = tokens.len() as u32;
        Ok(())
    }

    fn logits(&self) -> Result<&[f32]> {
        let all = self
            .arena
            .buffer
            .as_f32()
            .ok_or_else(|| anyhow!("arena is not mapped"))?;
        let start = self.arena.logits as usize;
        Ok(&all[start..start + self.config.n_vocab])
    }

    pub fn step(&mut self, token: u32) -> Result<&[f32]> {
        self.load(&[token])?;
        self.ctx.replay(&self.decode)?;
        self.pos += 1;
        self.logits()
    }

    pub fn prefill(&mut self, tokens: &[u32]) -> Result<&[f32]> {
        ensure!(!tokens.is_empty(), "prefill needs at least one token");
        for chunk in tokens.chunks(BATCH) {
            self.load(chunk)?;
            self.ctx.replay(&self.prefill)?;
            self.pos += chunk.len() as u32;
        }
        self.logits()
    }
}
