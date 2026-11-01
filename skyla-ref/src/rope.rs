use anyhow::{bail, Result};
use skyla_gguf::Gguf;

const BETA_FAST: f32 = 32.0;
const BETA_SLOW: f32 = 1.0;

#[derive(Debug, Clone)]
pub struct RopeScaling {
    pub factor: f32,
    pub original_ctx: f32,
    pub attn_factor: f32,
}

impl RopeScaling {
    pub fn from_gguf(g: &Gguf, arch: &str) -> Result<Option<Self>> {
        let k = |s: &str| format!("{arch}.rope.scaling.{s}");
        let kind = match g.meta_str(&k("type")) {
            Ok(t) => t,
            Err(_) => return Ok(None),
        };
        match kind {
            "none" => Ok(None),
            "yarn" => Ok(Some(Self {
                factor: g.meta_f64(&k("factor"))? as f32,
                original_ctx: g.meta_f64(&k("original_context_length"))? as f32,
                attn_factor: g.meta_f64(&k("attn_factor")).unwrap_or(1.0) as f32,
            })),
            other => bail!("unsupported rope scaling {other}"),
        }
    }
}

fn corr_dim(n_dims: f32, original_ctx: f32, n_rot: f32, base: f32) -> f32 {
    n_dims * (original_ctx / (n_rot * 2.0 * std::f32::consts::PI)).ln() / (2.0 * base.ln())
}

pub fn frequencies(head_dim: usize, theta: f32, scaling: Option<&RopeScaling>) -> (Vec<f32>, f32) {
    let d = head_dim as f32;
    let base: Vec<f32> = (0..head_dim / 2).map(|i| theta.powf(-2.0 * i as f32 / d)).collect();
    let Some(s) = scaling else {
        return (base, 1.0);
    };
    let freq_scale = 1.0 / s.factor;
    let low = corr_dim(d, s.original_ctx, BETA_FAST, theta).floor().max(0.0);
    let high = corr_dim(d, s.original_ctx, BETA_SLOW, theta).ceil().min(d - 1.0);
    let freqs = base
        .iter()
        .enumerate()
        .map(|(i, &f)| {
            let y = (i as f32 - low) / (high - low).max(0.001);
            let mix = 1.0 - y.clamp(0.0, 1.0);
            f * (freq_scale * (1.0 - mix) + mix)
        })
        .collect();
    let mscale = s.attn_factor * (1.0 + 0.1 * (1.0 / freq_scale).ln());
    (freqs, mscale)
}

pub fn rope_neox(x: &mut [f32], pos: usize, freqs: &[f32], mscale: f32) {
    let half = x.len() / 2;
    for (i, &f) in freqs.iter().enumerate().take(half) {
        let (sin, cos) = (pos as f32 * f).sin_cos();
        let (sin, cos) = (sin * mscale, cos * mscale);
        let (a, b) = (x[i], x[i + half]);
        x[i] = a * cos - b * sin;
        x[i + half] = a * sin + b * cos;
    }
}
