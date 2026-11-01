use crate::ops::{dot, rmsnorm, rmsnorm_inplace, silu, softmax};
use crate::{rope_neox, Config, Matrix};
use anyhow::Result;
use skyla_gguf::Gguf;

struct Layer<'a> {
    attn_norm: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    wq: Matrix<'a>,
    wk: Matrix<'a>,
    wv: Matrix<'a>,
    wo: Matrix<'a>,
    w_gate: Matrix<'a>,
    w_up: Matrix<'a>,
    w_down: Matrix<'a>,
}

impl<'a> Layer<'a> {
    fn load(g: &'a Gguf, i: usize) -> Result<Self> {
        let n = |s: &str| format!("blk.{i}.{s}.weight");
        Ok(Self {
            attn_norm: g.to_f32(&n("attn_norm"))?,
            q_norm: g.to_f32(&n("attn_q_norm"))?,
            k_norm: g.to_f32(&n("attn_k_norm"))?,
            ffn_norm: g.to_f32(&n("ffn_norm"))?,
            wq: Matrix::new(g, &n("attn_q"))?,
            wk: Matrix::new(g, &n("attn_k"))?,
            wv: Matrix::new(g, &n("attn_v"))?,
            wo: Matrix::new(g, &n("attn_output"))?,
            w_gate: Matrix::new(g, &n("ffn_gate"))?,
            w_up: Matrix::new(g, &n("ffn_up"))?,
            w_down: Matrix::new(g, &n("ffn_down"))?,
        })
    }
}

pub struct Model<'a> {
    pub config: Config,
    token_embd: Matrix<'a>,
    output_norm: Vec<f32>,
    output: Matrix<'a>,
    layers: Vec<Layer<'a>>,
    k_cache: Vec<Vec<f32>>,
    v_cache: Vec<Vec<f32>>,
    pub pos: usize,
}

impl<'a> Model<'a> {
    pub fn load(g: &'a Gguf) -> Result<Self> {
        let config = Config::from_gguf(g)?;
        let layers = (0..config.n_layer)
            .map(|i| Layer::load(g, i))
            .collect::<Result<Vec<_>>>()?;
        let token_embd = Matrix::new(g, "token_embd.weight")?;
        let output = match g.tensors.contains_key("output.weight") {
            true => Matrix::new(g, "output.weight")?,
            false => token_embd,
        };
        Ok(Self {
            k_cache: vec![Vec::new(); config.n_layer],
            v_cache: vec![Vec::new(); config.n_layer],
            config,
            token_embd,
            output_norm: g.to_f32("output_norm.weight")?,
            output,
            layers,
            pos: 0,
        })
    }

    pub fn reset(&mut self) {
        self.k_cache.iter_mut().for_each(Vec::clear);
        self.v_cache.iter_mut().for_each(Vec::clear);
        self.pos = 0;
    }

    pub fn forward(&mut self, token: u32) -> Vec<f32> {
        let c = self.config.clone();
        let hd = c.head_dim;
        let kv_dim = c.n_head_kv * hd;
        let mut x = vec![0.0f32; c.n_embd];
        self.token_embd.row(token as usize, &mut x);

        let mut n = vec![0.0f32; c.n_embd];
        let mut q = vec![0.0f32; c.n_head * hd];
        let mut k = vec![0.0f32; kv_dim];
        let mut v = vec![0.0f32; kv_dim];
        let mut attn = vec![0.0f32; c.n_head * hd];
        let mut o = vec![0.0f32; c.n_embd];
        let mut gate = vec![0.0f32; c.n_ff];
        let mut up = vec![0.0f32; c.n_ff];

        for l in 0..self.layers.len() {
            let layer = &self.layers[l];
            rmsnorm(&x, &layer.attn_norm, c.eps, &mut n);
            layer.wq.matvec(&n, &mut q);
            layer.wk.matvec(&n, &mut k);
            layer.wv.matvec(&n, &mut v);
            for h in q.chunks_mut(hd) {
                rmsnorm_inplace(h, &layer.q_norm, c.eps);
                rope_neox(h, self.pos, &c.rope_freqs, c.rope_mscale);
            }
            for h in k.chunks_mut(hd) {
                rmsnorm_inplace(h, &layer.k_norm, c.eps);
                rope_neox(h, self.pos, &c.rope_freqs, c.rope_mscale);
            }
            self.k_cache[l].extend_from_slice(&k);
            self.v_cache[l].extend_from_slice(&v);
            self.attention(l, &q, &mut attn);

            let layer = &self.layers[l];
            layer.wo.matvec(&attn, &mut o);
            x.iter_mut().zip(&o).for_each(|(x, o)| *x += o);
            rmsnorm(&x, &layer.ffn_norm, c.eps, &mut n);
            layer.w_gate.matvec(&n, &mut gate);
            layer.w_up.matvec(&n, &mut up);
            gate.iter_mut().zip(&up).for_each(|(g, u)| *g = silu(*g) * u);
            layer.w_down.matvec(&gate, &mut o);
            x.iter_mut().zip(&o).for_each(|(x, o)| *x += o);
        }

        rmsnorm(&x.clone(), &self.output_norm, c.eps, &mut x);
        let mut logits = vec![0.0f32; c.n_vocab];
        self.output.matvec(&x, &mut logits);
        self.pos += 1;
        logits
    }

    fn attention(&self, l: usize, q: &[f32], out: &mut [f32]) {
        let c = &self.config;
        let hd = c.head_dim;
        let kv_dim = c.n_head_kv * hd;
        let n_rep = c.n_head / c.n_head_kv;
        let scale = 1.0 / (hd as f32).sqrt();
        let mut scores = vec![0.0f32; self.pos + 1];
        for h in 0..c.n_head {
            let base = (h / n_rep) * hd;
            let qh = &q[h * hd..(h + 1) * hd];
            for (p, s) in scores.iter_mut().enumerate() {
                *s = dot(qh, &self.k_cache[l][p * kv_dim + base..][..hd]) * scale;
            }
            softmax(&mut scores);
            let oh = &mut out[h * hd..(h + 1) * hd];
            oh.fill(0.0);
            for (p, &w) in scores.iter().enumerate() {
                let vp = &self.v_cache[l][p * kv_dim + base..][..hd];
                oh.iter_mut().zip(vp).for_each(|(o, v)| *o += w * v);
            }
        }
    }
}
