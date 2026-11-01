use crate::arena::{Arena, ATTENTION_CHUNK};
use crate::kernels::{Kernels, ROWS_PER_GROUP};
use crate::weights::{QMatrix, Weights};
use anyhow::Result;
use skyla_ref::Config;
use skyla_vk::{Context, Pipeline, Recorder, Recording};

fn matvec(r: &Recorder, p: &Pipeline, m: QMatrix, x: u32, y: u32) {
    let groups = m.n_out.div_ceil(ROWS_PER_GROUP);
    r.dispatch(p, [groups, 1, 1], &[m.q, m.d, x, y, m.n_in, m.n_out]);
}

fn rmsnorm(r: &Recorder, k: &Kernels, x: u32, w: u32, y: u32, n: usize, eps: f32) {
    r.dispatch(&k.rmsnorm, [1, 1, 1], &[x, w, y, n as u32, eps.to_bits()]);
}

fn stage(r: &Recorder, label: &'static str) {
    r.barrier();
    r.mark(label);
}

pub fn record(ctx: &Context, k: &Kernels, w: &Weights, a: &Arena, c: &Config, profile: bool) -> Result<Recording> {
    let n_embd = c.n_embd as u32;
    let scale = 1.0 / (c.head_dim as f32).sqrt();
    ctx.record(profile, |r| {
        let push = [w.embd.q * 4, w.embd.d, a.x, n_embd];
        r.dispatch(&k.embed, [n_embd.div_ceil(256), 1, 1], &push);
        stage(r, "embed");

        for (l, lw) in w.layers.iter().enumerate() {
            rmsnorm(r, k, a.x, lw.attn_norm, a.n, c.n_embd, c.eps);
            stage(r, "rmsnorm");
            matvec(r, &k.matvec, lw.wq, a.n, a.q);
            matvec(r, &k.matvec, lw.wk, a.n, a.k);
            matvec(r, &k.matvec, lw.wv, a.n, a.v);
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
                c.rope_theta.to_bits(),
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
            matvec(r, &k.matvec_acc, lw.wo, a.attn, a.x);
            stage(r, "matvec o");

            rmsnorm(r, k, a.x, lw.ffn_norm, a.n, c.n_embd, c.eps);
            stage(r, "rmsnorm");
            matvec(r, &k.matvec, lw.gate, a.n, a.gate);
            matvec(r, &k.matvec, lw.up, a.n, a.up);
            stage(r, "matvec gate+up");
            let n_ff = c.n_ff as u32;
            r.dispatch(&k.silu, [n_ff.div_ceil(256), 1, 1], &[a.gate, a.up, n_ff]);
            stage(r, "silu");
            matvec(r, &k.matvec_acc, lw.down, a.gate, a.x);
            stage(r, "matvec down");
        }

        rmsnorm(r, k, a.x, w.out_norm, a.n, c.n_embd, c.eps);
        stage(r, "rmsnorm");
        matvec(r, &k.matvec, w.output, a.n, a.logits);
        r.host_barrier();
        r.mark("matvec output");
    })
}
