mod matrix;
mod model;
mod ops;
mod rope;

pub use matrix::Matrix;
pub use model::Model;
pub use ops::{argmax, rmsnorm, softmax};
pub use rope::{rope_neox, RopeScaling};

use anyhow::{ensure, Result};
use skyla_gguf::Gguf;

#[derive(Debug, Clone)]
pub struct Config {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub n_vocab: usize,
    pub rope_theta: f32,
    pub rope_scaling: Option<RopeScaling>,
    pub rope_freqs: Vec<f32>,
    pub rope_mscale: f32,
    pub eps: f32,
}

impl Config {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let arch = g.meta_str("general.architecture")?;
        ensure!(arch == "qwen3", "unsupported architecture {arch} (only qwen3)");
        let k = |s: &str| format!("{arch}.{s}");
        let n_embd = g.meta_u64(&k("embedding_length"))? as usize;
        let n_head = g.meta_u64(&k("attention.head_count"))? as usize;
        let head_dim = g
            .meta_u64(&k("attention.key_length"))
            .map_or(n_embd / n_head, |v| v as usize);
        let rope_theta = g.meta_f64(&k("rope.freq_base"))? as f32;
        let rope_scaling = RopeScaling::from_gguf(g, arch)?;
        let (rope_freqs, rope_mscale) = rope::frequencies(head_dim, rope_theta, rope_scaling.as_ref());
        Ok(Self {
            n_layer: g.meta_u64(&k("block_count"))? as usize,
            n_embd,
            n_head,
            n_head_kv: g.meta_u64(&k("attention.head_count_kv"))? as usize,
            head_dim,
            n_ff: g.meta_u64(&k("feed_forward_length"))? as usize,
            n_vocab: g.tensor("token_embd.weight")?.dims[1] as usize,
            rope_theta,
            rope_scaling,
            rope_freqs,
            rope_mscale,
            eps: g.meta_f64(&k("attention.layer_norm_rms_epsilon"))? as f32,
        })
    }
}
