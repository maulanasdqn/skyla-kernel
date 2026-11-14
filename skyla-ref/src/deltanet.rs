use crate::ops::{dot, l2norm_inplace, rmsnorm_inplace, sigmoid, silu, softplus};
use crate::{Matrix, Ssm};
use anyhow::Result;
use skyla_gguf::Gguf;

pub struct DeltaNet<'a> {
    qkv: Matrix<'a>,
    z: Matrix<'a>,
    alpha: Matrix<'a>,
    beta: Matrix<'a>,
    conv: Vec<f32>,
    a: Vec<f32>,
    dt: Vec<f32>,
    norm: Vec<f32>,
    out: Matrix<'a>,
    conv_state: Vec<f32>,
    state: Vec<f32>,
}

impl<'a> DeltaNet<'a> {
    pub fn load(g: &'a Gguf, i: usize, s: &Ssm) -> Result<Self> {
        let n = |t: &str| format!("blk.{i}.{t}.weight");
        Ok(Self {
            qkv: Matrix::new(g, &n("attn_qkv"))?,
            z: Matrix::new(g, &n("attn_gate"))?,
            alpha: Matrix::new(g, &n("ssm_alpha"))?,
            beta: Matrix::new(g, &n("ssm_beta"))?,
            conv: g.to_f32(&n("ssm_conv1d"))?,
            a: g.to_f32(&format!("blk.{i}.ssm_a"))?,
            dt: g.to_f32(&format!("blk.{i}.ssm_dt.bias"))?,
            norm: g.to_f32(&n("ssm_norm"))?,
            out: Matrix::new(g, &n("ssm_out"))?,
            conv_state: vec![0.0; s.channels() * (s.conv - 1)],
            state: vec![0.0; s.n_v_head * s.head_dim * s.head_dim],
        })
    }

    pub fn reset(&mut self) {
        self.conv_state.fill(0.0);
        self.state.fill(0.0);
    }

    fn convolve(&mut self, x: &mut [f32], s: &Ssm) {
        let hist = s.conv - 1;
        for (c, x) in x.iter_mut().enumerate() {
            let w = &self.conv[c * s.conv..(c + 1) * s.conv];
            let past = &mut self.conv_state[c * hist..(c + 1) * hist];
            let y = dot(&w[..hist], past) + w[hist] * *x;
            past.rotate_left(1);
            past[hist - 1] = *x;
            *x = silu(y);
        }
    }

    pub fn forward(&mut self, n: &[f32], s: &Ssm, eps: f32, out: &mut [f32]) {
        let (d, kd) = (s.head_dim, s.key_dim());
        let mut mixed = vec![0.0f32; s.channels()];
        let mut z = vec![0.0f32; s.value_dim()];
        let mut alpha = vec![0.0f32; s.n_v_head];
        let mut beta = vec![0.0f32; s.n_v_head];
        self.qkv.matvec(n, &mut mixed);
        self.z.matvec(n, &mut z);
        self.alpha.matvec(n, &mut alpha);
        self.beta.matvec(n, &mut beta);
        self.convolve(&mut mixed, s);

        let (qk, v) = mixed.split_at_mut(2 * kd);
        qk.chunks_mut(d).for_each(|h| l2norm_inplace(h, eps));
        let (q, k) = qk.split_at(kd);
        let scale = 1.0 / (d as f32).sqrt();
        let mut o = vec![0.0f32; s.value_dim()];
        let mut delta = vec![0.0f32; d];
        for h in 0..s.n_v_head {
            let kh = h % s.n_k_head;
            let (qh, kh) = (&q[kh * d..][..d], &k[kh * d..][..d]);
            let vh = &v[h * d..][..d];
            let b = sigmoid(beta[h]);
            let decay = (softplus(alpha[h] + self.dt[h]) * self.a[h]).exp();
            let m = &mut self.state[h * d * d..][..d * d];
            m.iter_mut().for_each(|x| *x *= decay);
            for (j, row) in m.chunks(d).enumerate() {
                delta[j] = (vh[j] - dot(row, kh)) * b;
            }
            for (row, dj) in m.chunks_mut(d).zip(&delta) {
                row.iter_mut().zip(kh).for_each(|(r, k)| *r += dj * k);
            }
            let oh = &mut o[h * d..][..d];
            for (oj, row) in oh.iter_mut().zip(m.chunks(d)) {
                *oj = dot(row, qh) * scale;
            }
            rmsnorm_inplace(oh, &self.norm, eps);
            oh.iter_mut().zip(&z[h * d..][..d]).for_each(|(o, z)| *o *= silu(*z));
        }
        self.out.matvec(&o, out);
    }
}
