use crate::arena::BATCH;
use crate::graph::{self, Mode};
use crate::Engine;
use anyhow::{anyhow, ensure, Result};

impl Engine<'_> {
    pub fn residuals(
        &mut self,
        tokens: &[u32],
        layers: &[usize],
        mut f: impl FnMut(usize, usize, &[f32]),
    ) -> Result<()> {
        ensure!(
            self.config.ssm.is_none(),
            "residual taps replay layers, which recurrent models cannot"
        );
        let recordings = layers
            .iter()
            .map(|&l| {
                let (k, w, a, c) = (&self.kernels, &self.weights, &self.arena, &self.config);
                graph::record(&self.ctx, k, w, a, c, Mode::Prefix(l + 1), false)
            })
            .collect::<Result<Vec<_>>>()?;
        let n = self.config.n_embd;
        let mut start = 0;
        for chunk in tokens.chunks(BATCH) {
            for (i, rec) in recordings.iter().enumerate() {
                self.load(chunk)?;
                self.ctx.replay(rec)?;
                let arena = self
                    .arena
                    .buffer
                    .as_f32()
                    .ok_or_else(|| anyhow!("arena is not mapped"))?;
                let (x, stride) = (self.arena.x as usize, self.arena.s_embd as usize);
                for t in 0..chunk.len() {
                    f(start + t, i, &arena[x + t * stride..x + t * stride + n]);
                }
            }
            self.load(chunk)?;
            self.ctx.replay(&self.prefill)?;
            self.pos += chunk.len() as u32;
            start += chunk.len();
        }
        Ok(())
    }
}
