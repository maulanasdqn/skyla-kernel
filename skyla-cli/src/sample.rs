pub struct Sampler {
    temp: f32,
    top_k: usize,
    top_p: f32,
    state: u64,
}

impl Sampler {
    pub fn new(temp: f32, top_k: usize, top_p: f32, seed: u64) -> Self {
        Self {
            temp,
            top_k: top_k.max(1),
            top_p,
            state: seed | 1,
        }
    }

    fn uniform(&mut self) -> f32 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        let bits = self.state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40;
        bits as f32 / (1u64 << 24) as f32
    }

    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        if self.temp <= 0.0 {
            return skyla_ref::argmax(logits);
        }
        let mut cand: Vec<(u32, f32)> = logits.iter().enumerate().map(|(i, &l)| (i as u32, l)).collect();
        let k = self.top_k.min(cand.len());
        cand.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
        cand.truncate(k);
        cand.sort_by(|a, b| b.1.total_cmp(&a.1));

        let max = cand[0].1;
        let mut probs: Vec<f32> = cand.iter().map(|c| ((c.1 - max) / self.temp).exp()).collect();
        let sum: f32 = probs.iter().sum();
        probs.iter_mut().for_each(|p| *p /= sum);

        let mut kept = 0.0;
        let mut cut = probs.len();
        for (i, p) in probs.iter().enumerate() {
            kept += p;
            if kept >= self.top_p {
                cut = i + 1;
                break;
            }
        }

        let mut r = self.uniform() * probs[..cut].iter().sum::<f32>();
        for (c, p) in cand.iter().zip(&probs[..cut]) {
            r -= p;
            if r <= 0.0 {
                return c.0;
            }
        }
        cand[cut - 1].0
    }
}
