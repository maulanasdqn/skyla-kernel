pub fn rmsnorm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let r = inv_rms(x, eps);
    out.iter_mut()
        .zip(x.iter().zip(w))
        .for_each(|(o, (x, w))| *o = x * r * w);
}

pub(crate) fn rmsnorm_inplace(x: &mut [f32], w: &[f32], eps: f32) {
    let r = inv_rms(x, eps);
    x.iter_mut().zip(w).for_each(|(x, w)| *x *= r * w);
}

fn inv_rms(x: &[f32], eps: f32) -> f32 {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    1.0 / (ms + eps).sqrt()
}

pub fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    x.iter_mut().for_each(|v| *v /= sum);
}

pub(crate) fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

pub fn argmax(x: &[f32]) -> u32 {
    x.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i as u32)
}
