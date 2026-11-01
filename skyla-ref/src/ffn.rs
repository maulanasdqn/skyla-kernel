use crate::ops::{dot, silu, softmax};
use crate::{Config, Matrix};
use anyhow::Result;
use skyla_gguf::Gguf;

pub enum Ffn<'a> {
    Dense {
        gate: Matrix<'a>,
        up: Matrix<'a>,
        down: Matrix<'a>,
    },
    Moe {
        router: Vec<f32>,
        gate: Matrix<'a>,
        up: Matrix<'a>,
        down: Matrix<'a>,
        used: usize,
    },
}

fn swiglu(gate: Matrix, up: Matrix, down: Matrix, n: &[f32], out: &mut [f32]) {
    let mut g = vec![0.0f32; gate.n_out];
    let mut u = vec![0.0f32; up.n_out];
    gate.matvec(n, &mut g);
    up.matvec(n, &mut u);
    g.iter_mut().zip(&u).for_each(|(g, u)| *g = silu(*g) * u);
    down.matvec(&g, out);
}

pub fn route(router: &[f32], n: &[f32], used: usize) -> Vec<(usize, f32)> {
    let mut logits: Vec<(usize, f32)> = router.chunks(n.len()).map(|r| dot(r, n)).enumerate().collect();
    logits.sort_by(|a, b| b.1.total_cmp(&a.1));
    logits.truncate(used);
    let mut weights: Vec<f32> = logits.iter().map(|l| l.1).collect();
    softmax(&mut weights);
    logits.iter().zip(weights).map(|(l, w)| (l.0, w)).collect()
}

impl<'a> Ffn<'a> {
    pub fn load(g: &'a Gguf, i: usize, c: &Config) -> Result<Self> {
        let n = |s: &str| format!("blk.{i}.{s}.weight");
        if c.n_expert == 0 {
            return Ok(Self::Dense {
                gate: Matrix::new(g, &n("ffn_gate"))?,
                up: Matrix::new(g, &n("ffn_up"))?,
                down: Matrix::new(g, &n("ffn_down"))?,
            });
        }
        Ok(Self::Moe {
            router: g.to_f32(&n("ffn_gate_inp"))?,
            gate: Matrix::new(g, &n("ffn_gate_exps"))?,
            up: Matrix::new(g, &n("ffn_up_exps"))?,
            down: Matrix::new(g, &n("ffn_down_exps"))?,
            used: c.n_expert_used,
        })
    }

    pub fn forward(&self, n: &[f32], out: &mut [f32]) {
        match self {
            Self::Dense { gate, up, down } => swiglu(*gate, *up, *down, n, out),
            Self::Moe {
                router,
                gate,
                up,
                down,
                used,
            } => {
                out.fill(0.0);
                let mut y = vec![0.0f32; out.len()];
                for (e, w) in route(router, n, *used) {
                    swiglu(gate.expert(e), up.expert(e), down.expert(e), n, &mut y);
                    out.iter_mut().zip(&y).for_each(|(o, y)| *o += w * y);
                }
            }
        }
    }
}
