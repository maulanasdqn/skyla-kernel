mod deltanet;
mod ffn;
mod matrix;
mod model;
mod ops;
mod rope;

pub use ffn::route;
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
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_ff_expert: usize,
    pub n_vocab: usize,
    pub rope_theta: f32,
    pub rope_scaling: Option<RopeScaling>,
    pub rope_freqs: Vec<f32>,
    pub rope_mscale: f32,
    pub n_rot: usize,
    pub eps: f32,
    pub attn_gate: bool,
    pub ssm: Option<Ssm>,
}

#[derive(Debug, Clone)]
pub struct Ssm {
    pub conv: usize,
    pub n_k_head: usize,
    pub n_v_head: usize,
    pub head_dim: usize,
    pub interval: usize,
}

impl Ssm {
    pub fn key_dim(&self) -> usize {
        self.n_k_head * self.head_dim
    }

    pub fn value_dim(&self) -> usize {
        self.n_v_head * self.head_dim
    }

    pub fn channels(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }

    fn from_gguf(g: &Gguf, arch: &str) -> Result<Option<Self>> {
        let k = |s: &str| format!("{arch}.{s}");
        if g.meta_u64(&k("ssm.state_size")).is_err() {
            return Ok(None);
        }
        let u = |s: &str| g.meta_u64(&k(s)).map(|v| v as usize);
        Ok(Some(Self {
            conv: u("ssm.conv_kernel")?,
            n_k_head: u("ssm.group_count")?,
            n_v_head: u("ssm.time_step_rank")?,
            head_dim: u("ssm.state_size")?,
            interval: u("full_attention_interval").unwrap_or(4),
        }))
    }
}

impl Config {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let arch = g.meta_str("general.architecture")?;
        ensure!(
            matches!(arch, "qwen3" | "qwen3moe" | "qwen3vl" | "qwen35"),
            "unsupported architecture {arch} (qwen3, qwen3moe, qwen3vl and qwen35 text only)"
        );
        let k = |s: &str| format!("{arch}.{s}");
        let opt = |s: &str| g.meta_u64(&k(s)).map_or(0, |v| v as usize);
        let n_embd = g.meta_u64(&k("embedding_length"))? as usize;
        let n_head = g.meta_u64(&k("attention.head_count"))? as usize;
        let head_dim = g
            .meta_u64(&k("attention.key_length"))
            .map_or(n_embd / n_head, |v| v as usize);
        let rope_theta = g.meta_f64(&k("rope.freq_base"))? as f32;
        let n_rot = match opt("rope.dimension_count") {
            0 => head_dim,
            n => n,
        };
        let rope_scaling = RopeScaling::from_gguf(g, arch)?;
        let (rope_freqs, rope_mscale) = rope::frequencies(n_rot, rope_theta, rope_scaling.as_ref());
        Ok(Self {
            n_layer: g.meta_u64(&k("block_count"))? as usize - opt("nextn_predict_layers"),
            n_embd,
            n_head,
            n_head_kv: g.meta_u64(&k("attention.head_count_kv"))? as usize,
            head_dim,
            n_ff: opt("feed_forward_length"),
            n_expert: opt("expert_count"),
            n_expert_used: opt("expert_used_count"),
            n_ff_expert: opt("expert_feed_forward_length"),
            n_vocab: g.tensor("token_embd.weight")?.dims[1] as usize,
            rope_theta,
            rope_scaling,
            rope_freqs,
            rope_mscale,
            n_rot,
            eps: g.meta_f64(&k("attention.layer_norm_rms_epsilon"))? as f32,
            attn_gate: arch == "qwen35",
            ssm: Ssm::from_gguf(g, arch)?,
        })
    }

    pub fn recurrent(&self, layer: usize) -> bool {
        self.ssm.as_ref().is_some_and(|s| !(layer + 1).is_multiple_of(s.interval))
    }

    pub fn ffn_norm(&self) -> &'static str {
        match self.ssm {
            Some(_) => "post_attention_norm",
            None => "ffn_norm",
        }
    }
}
