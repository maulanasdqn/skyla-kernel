use skyla_types::Settings;
use std::collections::VecDeque;

const REPEAT_WINDOW: usize = 64;
const REPEAT_PENALTY: f32 = 1.1;

pub struct Sampler {
    temp: f32,
    top_k: usize,
    top_p: f32,
    state: u64,
    recent: VecDeque<u32>,
}

impl Sampler {
    pub fn new(temp: f32, top_k: usize, top_p: f32, seed: u64) -> Self {
        Self {
            temp,
            top_k: top_k.max(1),
            top_p,
            state: seed | 1,
            recent: VecDeque::with_capacity(REPEAT_WINDOW),
        }
    }

    pub fn from_settings(s: &Settings, seed: u64) -> Self {
        Self::new(s.temperature, s.top_k, s.top_p, seed)
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
        let token = self.pick(logits);
        if self.recent.len() == REPEAT_WINDOW {
            self.recent.pop_front();
        }
        self.recent.push_back(token);
        token
    }

    fn penalized(&self, logits: &[f32]) -> Vec<(u32, f32)> {
        let mut cand: Vec<(u32, f32)> = logits.iter().enumerate().map(|(i, &l)| (i as u32, l)).collect();
        let mut seen: Vec<u32> = self.recent.iter().copied().collect();
        seen.sort_unstable();
        seen.dedup();
        for t in seen {
            let l = &mut cand[t as usize].1;
            *l = if *l > 0.0 {
                *l / REPEAT_PENALTY
            } else {
                *l * REPEAT_PENALTY
            };
        }
        cand
    }

    fn pick(&mut self, logits: &[f32]) -> u32 {
        let mut cand = self.penalized(logits);
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

pub fn seed_from_clock() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(1, |d| d.as_nanos() as u64)
}
