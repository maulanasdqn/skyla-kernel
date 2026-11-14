use crate::deltanet::DeltaNet;
use crate::ffn::Ffn;
use crate::ops::{dot, rmsnorm, rmsnorm_inplace, sigmoid, softmax};
use crate::{rope_neox, Config, Matrix};
use anyhow::Result;
use skyla_gguf::Gguf;

struct Attention<'a> {
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    wq: Matrix<'a>,
    wk: Matrix<'a>,
    wv: Matrix<'a>,
    wo: Matrix<'a>,
}

enum Mixer<'a> {
    Attention(Attention<'a>),
    DeltaNet(DeltaNet<'a>),
}

struct Layer<'a> {
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    mixer: Mixer<'a>,
    ffn: Ffn<'a>,
}

impl<'a> Layer<'a> {
    fn load(g: &'a Gguf, i: usize, c: &Config) -> Result<Self> {
        let n = |s: &str| format!("blk.{i}.{s}.weight");
        let mixer = match c.ssm.as_ref().filter(|_| c.recurrent(i)) {
            Some(s) => Mixer::DeltaNet(DeltaNet::load(g, i, s)?),
            None => Mixer::Attention(Attention {
                q_norm: g.to_f32(&n("attn_q_norm"))?,
                k_norm: g.to_f32(&n("attn_k_norm"))?,
                wq: Matrix::new(g, &n("attn_q"))?,
                wk: Matrix::new(g, &n("attn_k"))?,
                wv: Matrix::new(g, &n("attn_v"))?,
                wo: Matrix::new(g, &n("attn_output"))?,
            }),
        };
        Ok(Self {
            attn_norm: g.to_f32(&n("attn_norm"))?,
            ffn_norm: g.to_f32(&n(c.ffn_norm()))?,
            mixer,
            ffn: Ffn::load(g, i, c)?,
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
            .map(|i| Layer::load(g, i, &config))
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
        for layer in &mut self.layers {
            if let Mixer::DeltaNet(d) = &mut layer.mixer {
                d.reset();
            }
        }
        self.pos = 0;
    }

    pub fn forward(&mut self, token: u32) -> Vec<f32> {
        let c = self.config.clone();
        let mut x = vec![0.0f32; c.n_embd];
        self.token_embd.row(token as usize, &mut x);

        let mut n = vec![0.0f32; c.n_embd];
        let mut o = vec![0.0f32; c.n_embd];
        for l in 0..self.layers.len() {
            rmsnorm(&x, &self.layers[l].attn_norm, c.eps, &mut n);
            match &mut self.layers[l].mixer {
                Mixer::DeltaNet(d) => d.forward(&n, c.ssm.as_ref().expect("recurrent layer"), c.eps, &mut o),
                Mixer::Attention(_) => self.attention(l, &n, &mut o),
            }
            x.iter_mut().zip(&o).for_each(|(x, o)| *x += o);
            let layer = &self.layers[l];
            rmsnorm(&x, &layer.ffn_norm, c.eps, &mut n);
            layer.ffn.forward(&n, &mut o);
            x.iter_mut().zip(&o).for_each(|(x, o)| *x += o);
        }

        rmsnorm(&x.clone(), &self.output_norm, c.eps, &mut x);
        let mut logits = vec![0.0f32; c.n_vocab];
        self.output.matvec(&x, &mut logits);
        self.pos += 1;
        logits
    }

    fn attention(&mut self, l: usize, n: &[f32], out: &mut [f32]) {
        let c = &self.config;
        let Mixer::Attention(at) = &self.layers[l].mixer else {
            unreachable!("attention on a recurrent layer");
        };
        let hd = c.head_dim;
        let kv_dim = c.n_head_kv * hd;
        let q_width = if c.attn_gate { 2 * hd } else { hd };
        let mut qg = vec![0.0f32; c.n_head * q_width];
        let mut k = vec![0.0f32; kv_dim];
        let mut v = vec![0.0f32; kv_dim];
        at.wq.matvec(n, &mut qg);
        at.wk.matvec(n, &mut k);
        at.wv.matvec(n, &mut v);
        for h in qg.chunks_mut(q_width) {
            rmsnorm_inplace(&mut h[..hd], &at.q_norm, c.eps);
            rope_neox(&mut h[..c.n_rot], self.pos, &c.rope_freqs, c.rope_mscale);
        }
        for h in k.chunks_mut(hd) {
            rmsnorm_inplace(h, &at.k_norm, c.eps);
            rope_neox(&mut h[..c.n_rot], self.pos, &c.rope_freqs, c.rope_mscale);
        }
        self.k_cache[l].extend_from_slice(&k);
        self.v_cache[l].extend_from_slice(&v);

        let n_rep = c.n_head / c.n_head_kv;
        let scale = 1.0 / (hd as f32).sqrt();
        let mut attn = vec![0.0f32; c.n_head * hd];
        let mut scores = vec![0.0f32; self.pos + 1];
        for h in 0..c.n_head {
            let base = (h / n_rep) * hd;
            let qh = &qg[h * q_width..][..hd];
            for (p, s) in scores.iter_mut().enumerate() {
                *s = dot(qh, &self.k_cache[l][p * kv_dim + base..][..hd]) * scale;
            }
            softmax(&mut scores);
            let oh = &mut attn[h * hd..(h + 1) * hd];
            oh.fill(0.0);
            for (p, &w) in scores.iter().enumerate() {
                let vp = &self.v_cache[l][p * kv_dim + base..][..hd];
                oh.iter_mut().zip(vp).for_each(|(o, v)| *o += w * v);
            }
            if c.attn_gate {
                let gate = &qg[h * q_width + hd..][..hd];
                oh.iter_mut().zip(gate).for_each(|(o, g)| *o *= sigmoid(*g));
            }
        }
        at.wo.matvec(&attn, out);
    }
}
