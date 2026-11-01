use crate::arena::Arena;
use crate::graph;
use crate::kernels::Kernels;
use crate::weights::Weights;
use anyhow::{anyhow, ensure, Result};
use skyla_gguf::Gguf;
use skyla_ref::Config;
use skyla_vk::{Context, Recording};

pub struct Engine {
    recording: Recording,
    _kernels: Kernels,
    arena: Arena,
    weights: Weights,
    pub config: Config,
    pub pos: u32,
    ctx: Context,
}

impl Engine {
    pub fn new(g: &Gguf, max_ctx: usize, profile: bool) -> Result<Self> {
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
        let ctx = Context::new()?;
        let weights = Weights::load(&ctx, g, config.n_layer)?;
        let arena = Arena::new(&ctx, &config, max_ctx)?;
        let n_rep = (config.n_head / config.n_head_kv) as u32;
        let kernels = Kernels::new(&ctx, &weights.buffer, &arena.buffer, &arena.kv, n_rep)?;
        let recording = graph::record(&ctx, &kernels, &weights, &arena, &config, profile)?;
        Ok(Self {
            recording,
            _kernels: kernels,
            arena,
            weights,
            config,
            pos: 0,
            ctx,
        })
    }

    pub fn device_name(&self) -> String {
        self.ctx.device_name()
    }

    pub fn weight_bytes(&self) -> u64 {
        self.weights.buffer.size
    }

    pub fn reset(&mut self) {
        self.pos = 0;
    }

    pub fn last_step_profile(&self) -> Result<Vec<(&'static str, f64)>> {
        self.ctx.marks(&self.recording)
    }

    pub fn step(&mut self, token: u32) -> Result<&[f32]> {
        ensure!(
            self.pos < self.arena.max_ctx,
            "context full ({} tokens)",
            self.arena.max_ctx
        );
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
