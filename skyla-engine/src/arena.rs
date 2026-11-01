use crate::host_memory;
use anyhow::{ensure, Result};
use skyla_ref::Config;
use skyla_vk::{Buffer, BufferUsageFlags, Context};

pub struct Arena {
    pub buffer: Buffer,
    pub kv: Buffer,
    pub x: u32,
    pub n: u32,
    pub q: u32,
    pub k: u32,
    pub v: u32,
    pub attn: u32,
    pub gate: u32,
    pub up: u32,
    pub logits: u32,
    pub part: u32,
    pub n_split: u32,
    pub max_ctx: u32,
    kv_dim: u32,
}

pub const ATTENTION_CHUNK: u32 = 64;

impl Arena {
    pub fn new(ctx: &Context, c: &Config, max_ctx: usize) -> Result<Self> {
        let mut size = 64usize;
        let mut region = |n: usize| {
            let off = size;
            size = (size + n).next_multiple_of(64);
            off as u32
        };
        let x = region(c.n_embd);
        let n = region(c.n_embd);
        let q = region(c.n_head * c.head_dim);
        let k = region(c.n_head_kv * c.head_dim);
        let v = region(c.n_head_kv * c.head_dim);
        let attn = region(c.n_head * c.head_dim);
        let gate = region(c.n_ff);
        let up = region(c.n_ff);
        let logits = region(c.n_vocab);
        let n_split = (max_ctx as u32).div_ceil(ATTENTION_CHUNK);
        let part = region(c.n_head * n_split as usize * 130);

        let memory = host_memory(ctx)?;
        let usage = BufferUsageFlags::STORAGE_BUFFER;
        let buffer = ctx.alloc_buffer(size as u64 * 4, usage, memory)?;
        let kv_dim = c.n_head_kv * c.head_dim;
        let kv_bytes = (c.n_layer * 2 * max_ctx * kv_dim * 2) as u64;
        ensure!(
            kv_bytes <= ctx.max_allocation_size,
            "KV cache for {max_ctx} tokens needs {} MiB, above the allocation cap",
            kv_bytes >> 20
        );
        let kv = ctx.alloc_buffer(kv_bytes, usage, memory)?;
        Ok(Self {
            buffer,
            kv,
            x,
            n,
            q,
            k,
            v,
            attn,
            gate,
            up,
            logits,
            part,
            n_split,
            max_ctx: max_ctx as u32,
            kv_dim: kv_dim as u32,
        })
    }

    pub fn k_cache(&self, layer: usize) -> u32 {
        layer as u32 * 2 * self.max_ctx * self.kv_dim
    }

    pub fn v_cache(&self, layer: usize) -> u32 {
        self.k_cache(layer) + self.max_ctx * self.kv_dim
    }
}
