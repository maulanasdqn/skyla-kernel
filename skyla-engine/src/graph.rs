use crate::arena::{Arena, ATTENTION_CHUNK, BATCH};
use crate::kernels::Kernels;
use crate::weights::{LayerWeights, QMatrix, Weights};
use anyhow::Result;
use skyla_gguf::DType;
use skyla_ref::Config;
use skyla_vk::{Context, Recorder, Recording};

struct Graph<'a> {
    r: &'a Recorder<'a>,
    k: &'a Kernels,
    a: &'a Arena,
    c: &'a Config,
    batched: bool,
    b: u32,
}

impl Graph<'_> {
    fn matvec(&self, m: QMatrix, x: (u32, u32), y: (u32, u32), accum: bool, batched: bool) {
        let mv = self.k.matvec(m.dtype, batched);
        let groups = m.n_out.div_ceil(mv.rows_per_group);
        let o = m.off;
        let push = match (batched, m.dtype) {
            (true, _) => vec![o[0], o[1], o[2], o[3], x.0, y.0, m.n_in, m.n_out, x.1, y.1],
            (false, DType::Q6K) => vec![o[0], o[1], o[2], o[3], x.0, y.0, m.n_in, m.n_out, 0, 0, 0],
            (false, _) => vec![o[0], o[1], x.0, y.0, m.n_in, m.n_out, 0, 0, 0],
        };
        self.r.dispatch_set(mv.pipeline(accum), m.buf, [groups, 1, 1], &push);
    }

    fn experts(&self, m: QMatrix, x: (u32, u32), y: (u32, u32)) {
        let units = match m.dtype {
            DType::Q4K => m.n_in / 32,
            DType::Q6K => m.n_in / 64,
            _ => 64,
        };
        let (pipeline, rows_per_group) = self.k.matvec(m.dtype, false).experts(units);
        let groups = m.n_out.div_ceil(rows_per_group);
        let (o, ids) = (m.off, self.a.moe.ids);
        let mut push = match m.dtype {
            DType::Q6K => vec![o[0], o[1], o[2], o[3]],
            _ => vec![o[0], o[1]],
        };
        push.extend([x.0, y.0, m.n_in, m.n_out, ids, x.1, y.1]);
        let slots = self.c.n_expert_used as u32;
        self.r.dispatch_set(pipeline, m.buf, [groups, slots, 1], &push);
    }

    fn moe(&self, lw: &LayerWeights, router: u32) {
        let (a, c, m) = (self.a, self.c, self.a.moe);
        let used = c.n_expert_used as u32;
        let n_embd = c.n_embd as u32;
        let n_expert = c.n_expert as u32;
        let push = [a.n, router, m.logits, n_embd];
        self.r.dispatch(&self.k.logits, [n_expert, 1, 1], &push);
        self.stage("moe logits");
        let push = [m.logits, m.ids, m.weights, n_expert, used];
        self.r.dispatch(&self.k.route, [1, 1, 1], &push);
        self.stage("moe route");
        self.experts(lw.gate, (a.n, 0), (m.gate, m.stride));
        self.experts(lw.up, (a.n, 0), (m.up, m.stride));
        self.stage("experts gate+up");
        let n = used * m.stride;
        self.r
            .dispatch(&self.k.silu, [n.div_ceil(256), 1, 1], &[m.gate, m.up, n, 0]);
        self.stage("silu");
        self.experts(lw.down, (m.gate, m.stride), (m.down, a.s_embd));
        self.stage("experts down");
        let push = [a.x, m.down, m.weights, n_embd, a.s_embd, used];
        self.r.dispatch(&self.k.combine, [n_embd.div_ceil(256), 1, 1], &push);
        self.stage("moe combine");
    }

    fn rmsnorm(&self, x: u32, w: u32, y: u32, tokens: u32) {
        let c = self.c;
        let push = [x, w, y, c.n_embd as u32, c.eps.to_bits(), self.a.s_embd];
        self.r.dispatch(&self.k.rmsnorm, [1, tokens, 1], &push);
    }

    fn stage(&self, label: &'static str) {
        self.r.barrier();
        self.r.mark(label);
    }

    fn layer(&self, l: usize, lw: &LayerWeights, w: &Weights) {
        let (a, c, b, bt) = (self.a, self.c, self.b, self.batched);
        let emb = |off| (off, a.s_embd);
        self.rmsnorm(a.x, lw.attn_norm, a.n, b);
        self.stage("rmsnorm");
        self.matvec(lw.wq, emb(a.n), (a.q, a.s_q), false, bt);
        self.matvec(lw.wk, emb(a.n), (a.k, a.s_kv), false, bt);
        self.matvec(lw.wv, emb(a.n), (a.v, a.s_kv), false, bt);
        self.stage("matvec qkv");
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
            a.s_q,
            a.s_kv,
            a.k_scales(l),
            a.v_scales(l),
        ];
        self.r.dispatch(&self.k.qkv, [heads, b, 1], &push);
        self.stage("qk norm + rope");
        let scale = 1.0 / (c.head_dim as f32).sqrt();
        let push = [
            a.q,
            a.part,
            a.k_cache(l),
            a.v_cache(l),
            c.n_head_kv as u32,
            a.n_split,
            scale.to_bits(),
            a.s_q,
            a.s_part,
            a.k_scales(l),
            a.v_scales(l),
        ];
        self.r
            .dispatch(&self.k.attention, [c.n_head_kv as u32, a.n_split, b], &push);
        self.stage("attention");
        let push = [a.part, a.attn, a.n_split, ATTENTION_CHUNK, a.s_part, a.s_q];
        self.r.dispatch(&self.k.merge, [c.n_head as u32, b, 1], &push);
        self.stage("attention merge");
        self.matvec(lw.wo, (a.attn, a.s_q), emb(a.x), true, bt);
        self.stage("matvec o");

        self.rmsnorm(a.x, lw.ffn_norm, a.n, b);
        self.stage("rmsnorm");
        if let Some(router) = lw.router {
            self.moe(lw, router);
            return;
        }
        self.matvec(lw.gate, emb(a.n), (a.gate, a.s_ff), false, bt);
        self.matvec(lw.up, emb(a.n), (a.up, a.s_ff), false, bt);
        self.stage("matvec gate+up");
        let n_ff = c.n_ff as u32;
        let push = [a.gate, a.up, n_ff, a.s_ff];
        self.r.dispatch(&self.k.silu, [n_ff.div_ceil(256), b, 1], &push);
        self.stage("silu");
        self.matvec(lw.down, (a.gate, a.s_ff), emb(a.x), true, bt);
        self.stage("matvec down");
    }
}

pub fn record(
    ctx: &Context,
    k: &Kernels,
    w: &Weights,
    a: &Arena,
    c: &Config,
    batched: bool,
    profile: bool,
) -> Result<Recording> {
    ctx.record(profile, |r| {
        let b = if batched { BATCH as u32 } else { 1 };
        let g = Graph { r, k, a, c, batched, b };
        for (l, lw) in w.layers.iter().enumerate() {
            g.layer(l, lw, w);
        }
        let src = match batched {
            true => {
                let n = c.n_embd as u32;
                r.dispatch(&k.take_last, [n.div_ceil(256), 1, 1], &[a.x, a.last, n, a.s_embd]);
                g.stage("take last");
                a.last
            }
            false => a.x,
        };
        g.rmsnorm(src, w.out_norm, a.last, 1);
        g.stage("rmsnorm");
        g.matvec(w.output, (a.last, 0), (a.logits, 0), false, false);
        r.host_barrier();
        r.mark("matvec output");
    })
}
