use crate::host_memory;
use crate::kv::{KvLayout, KvPrecision};
use anyhow::{ensure, Result};
use skyla_ref::Config;
use skyla_vk::{Buffer, BufferUsageFlags, Context};

pub const ATTENTION_CHUNK: u32 = 64;
pub const BATCH: usize = 128;

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
    pub part: u32,
    pub last: u32,
    pub logits: u32,
    pub s_embd: u32,
    pub s_q: u32,
    pub s_kv: u32,
    pub s_ff: u32,
    pub s_part: u32,
    pub n_split: u32,
    pub max_ctx: u32,
    pub kv_layout: KvLayout,
    pub moe: MoeRegions,
}

#[derive(Clone, Copy)]
pub struct MoeRegions {
    pub logits: u32,
    pub ids: u32,
    pub weights: u32,
    pub gate: u32,
    pub up: u32,
    pub down: u32,
    pub table: u32,
    pub list: u32,
    pub stride: u32,
}

struct Layout {
    size: usize,
}

impl Layout {
    fn batched(&mut self, width: usize) -> (u32, u32) {
        let stride = width.next_multiple_of(64);
        let off = self.size;
        self.size += stride * BATCH;
        (off as u32, stride as u32)
    }

    fn single(&mut self, width: usize) -> u32 {
        let off = self.size;
        self.size += width.next_multiple_of(64);
        off as u32
    }
}

impl Arena {
    pub fn new(ctx: &Context, c: &Config, max_ctx: usize, precision: KvPrecision) -> Result<Self> {
        let n_split = (max_ctx as u32).div_ceil(ATTENTION_CHUNK);
        let mut l = Layout { size: 64 };
        let (x, s_embd) = l.batched(c.n_embd);
        let (n, _) = l.batched(c.n_embd);
        let (q, s_q) = l.batched(c.n_head * c.head_dim);
        let (k, s_kv) = l.batched(c.n_head_kv * c.head_dim);
        let (v, _) = l.batched(c.n_head_kv * c.head_dim);
        let (attn, _) = l.batched(c.n_head * c.head_dim);
        let (gate, s_ff) = l.batched(c.n_ff);
        let (up, _) = l.batched(c.n_ff);
        let (part, s_part) = l.batched(c.n_head * n_split as usize * 130);
        let last = l.single(c.n_embd);
        let logits = l.single(c.n_vocab);
        let stride = c.n_ff_expert.next_multiple_of(64);
        let slots = c.n_expert_used * BATCH;
        let moe = MoeRegions {
            logits: l.single(c.n_expert * BATCH),
            ids: l.single(slots),
            weights: l.single(slots),
            gate: l.single(stride * slots),
            up: l.single(stride * slots),
            down: l.single(s_embd as usize * slots),
            table: l.single(2 * c.n_expert),
            list: l.single(slots),
            stride: stride as u32,
        };

        let memory = host_memory(ctx)?;
        let usage = BufferUsageFlags::STORAGE_BUFFER;
        let buffer = ctx.alloc_buffer(l.size as u64 * 4, usage, memory)?;
        let kv_layout = KvLayout::new(precision, c.n_layer, max_ctx, c.n_head_kv * c.head_dim);
        let kv_bytes = kv_layout.bytes();
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
            part,
            last,
            logits,
            s_embd,
            s_q,
            s_kv,
            s_ff,
            s_part,
            n_split,
            max_ctx: max_ctx as u32,
            kv_layout,
            moe,
        })
    }

    pub fn k_cache(&self, layer: usize) -> u32 {
        self.kv_layout.keys(layer)
    }

    pub fn v_cache(&self, layer: usize) -> u32 {
        self.kv_layout.values(layer)
    }

    pub fn k_scales(&self, layer: usize) -> u32 {
        self.kv_layout.key_scales(layer)
    }

    pub fn v_scales(&self, layer: usize) -> u32 {
        self.kv_layout.value_scales(layer)
    }
}
