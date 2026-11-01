use crate::arena::{Arena, ATTENTION_CHUNK};
use crate::kernels::{Kernels, ROWS_PER_GROUP};
use crate::weights::{QMatrix, Weights};
use anyhow::Result;
use skyla_gguf::DType;
use skyla_ref::Config;
use skyla_vk::{Context, Recorder, Recording};

fn matvec(r: &Recorder, k: &Kernels, m: QMatrix, x: u32, y: u32, accum: bool) {
    let groups = m.n_out.div_ceil(ROWS_PER_GROUP);
    let o = m.off;
    let push = match m.dtype {
        DType::Q6K => vec![o[0], o[1], o[2], o[3], x, y, m.n_in, m.n_out],
        _ => vec![o[0], o[1], x, y, m.n_in, m.n_out],
    };
    r.dispatch_set(k.matvec(m.dtype, accum), m.buf, [groups, 1, 1], &push);
}

fn rmsnorm(r: &Recorder, k: &Kernels, x: u32, w: u32, y: u32, n: usize, eps: f32) {
    r.dispatch(&k.rmsnorm, [1, 1, 1], &[x, w, y, n as u32, eps.to_bits()]);
}

fn stage(r: &Recorder, label: &'static str) {
    r.barrier();
    r.mark(label);
}

pub fn record(ctx: &Context, k: &Kernels, w: &Weights, a: &Arena, c: &Config, profile: bool) -> Result<Recording> {
    let scale = 1.0 / (c.head_dim as f32).sqrt();
    ctx.record(profile, |r| {
        for (l, lw) in w.layers.iter().enumerate() {
            rmsnorm(r, k, a.x, lw.attn_norm, a.n, c.n_embd, c.eps);
            stage(r, "rmsnorm");
            matvec(r, k, lw.wq, a.n, a.q, false);
            matvec(r, k, lw.wk, a.n, a.k, false);
            matvec(r, k, lw.wv, a.n, a.v, false);
            stage(r, "matvec qkv");
            let heads = (c.n_head + 2 * c.n_head_kv) as u32;
            let push = [
                a.q,
                a.k,
                a.v,
                lw.q_norm,
                lw.k_norm,
                a.k_cache(l),
                a.v_cache(l),
                c.n_head as u32,
                c.n_head_kv as u32,
                c.eps.to_bits(),
                w.rope,
                c.rope_mscale.to_bits(),
            ];
            r.dispatch(&k.qkv, [heads, 1, 1], &push);
            stage(r, "qk norm + rope");
            let push = [
                a.q,
                a.part,
                a.k_cache(l),
                a.v_cache(l),
                c.n_head_kv as u32,
                a.n_split,
                scale.to_bits(),
            ];
            r.dispatch(&k.attention, [c.n_head_kv as u32, a.n_split, 1], &push);
            stage(r, "attention");
            let push = [a.part, a.attn, a.n_split, ATTENTION_CHUNK];
            r.dispatch(&k.merge, [c.n_head as u32, 1, 1], &push);
            stage(r, "attention merge");
            matvec(r, k, lw.wo, a.attn, a.x, true);
            stage(r, "matvec o");

            rmsnorm(r, k, a.x, lw.ffn_norm, a.n, c.n_embd, c.eps);
            stage(r, "rmsnorm");
            matvec(r, k, lw.gate, a.n, a.gate, false);
            matvec(r, k, lw.up, a.n, a.up, false);
            stage(r, "matvec gate+up");
            let n_ff = c.n_ff as u32;
            r.dispatch(&k.silu, [n_ff.div_ceil(256), 1, 1], &[a.gate, a.up, n_ff]);
            stage(r, "silu");
            matvec(r, k, lw.down, a.gate, a.x, true);
            stage(r, "matvec down");
        }

        rmsnorm(r, k, a.x, w.out_norm, a.n, c.n_embd, c.eps);
        stage(r, "rmsnorm");
        matvec(r, k, w.output, a.n, a.logits, false);
        r.host_barrier();
        r.mark("matvec output");
    })
}
