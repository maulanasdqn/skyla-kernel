use crate::weights::QMatrix;
use anyhow::{bail, ensure, Result};
use skyla_gguf::{DType, Gguf};
use std::ops::Range;

type Part = (Range<usize>, u32);

fn layout(dtype: DType) -> Result<(usize, Vec<Part>)> {
    Ok(match dtype {
        DType::Q8_0 => (34, vec![(2..34, 16), (0..2, 2)]),
        DType::Q4K => (144, vec![(16..144, 16), (0..16, 16)]),
        DType::Q6K => (210, vec![(0..128, 16), (128..192, 16), (192..208, 16), (208..210, 2)]),
        other => bail!("no GPU kernel for {other:?} weights"),
    })
}

struct Split<'a> {
    buf: usize,
    src: &'a [u8],
    block: usize,
    dst: Vec<(Range<usize>, usize)>,
}

pub struct Planner<'a> {
    g: &'a Gguf,
    cap: usize,
    pub sizes: Vec<usize>,
    tables: Vec<(usize, Vec<f32>)>,
    splits: Vec<Split<'a>>,
}

impl<'a> Planner<'a> {
    pub fn new(g: &'a Gguf, cap: usize) -> Self {
        Self {
            g,
            cap,
            sizes: vec![0],
            tables: Vec::new(),
            splits: Vec::new(),
        }
    }

    fn alloc(&mut self, buf: usize, bytes: usize) -> usize {
        let off = self.sizes[buf].next_multiple_of(16);
        self.sizes[buf] = off + bytes;
        off
    }

    pub fn table(&mut self, data: Vec<f32>) -> u32 {
        let off = self.alloc(0, data.len() * 4);
        self.tables.push((off, data));
        (off / 4) as u32
    }

    pub fn norm(&mut self, name: &str) -> Result<u32> {
        let data = self.g.to_f32(name)?;
        Ok(self.table(data))
    }

    pub fn matrix(&mut self, name: &str) -> Result<QMatrix> {
        let t = self.g.tensor(name)?;
        let (n_in, n_out) = (t.dims[0] as usize, t.dims[1] as usize);
        ensure!(
            n_in % 256 == 0 || t.dtype == DType::Q8_0,
            "{name}: unsupported input size {n_in}"
        );
        let (block, parts) = layout(t.dtype)?;
        let n_blocks = t.n_bytes / block;
        let total = n_blocks * block + parts.len() * 16;
        let last = self.sizes.len() - 1;
        let buf = match self.sizes[last] + total <= self.cap {
            true => last,
            false => {
                self.sizes.push(0);
                last + 1
            }
        };
        let mut off = [0u32; 4];
        let mut dst = Vec::new();
        for (i, (range, unit)) in parts.into_iter().enumerate() {
            let start = self.alloc(buf, n_blocks * range.len());
            off[i] = (start / unit as usize) as u32;
            dst.push((range, start));
        }
        self.splits.push(Split {
            buf,
            src: self.g.data(t),
            block,
            dst,
        });
        Ok(QMatrix {
            dtype: t.dtype,
            buf,
            off,
            n_in: n_in as u32,
            n_out: n_out as u32,
        })
    }

    pub fn fill(&self, buf: usize, out: &mut [u8]) {
        for (off, data) in self.tables.iter().filter(|_| buf == 0) {
            for (i, x) in data.iter().enumerate() {
                out[off + i * 4..off + i * 4 + 4].copy_from_slice(&x.to_le_bytes());
            }
        }
        for s in self.splits.iter().filter(|s| s.buf == buf) {
            for (i, b) in s.src.chunks_exact(s.block).enumerate() {
                for (range, start) in &s.dst {
                    let at = start + i * range.len();
                    out[at..at + range.len()].copy_from_slice(&b[range.clone()]);
                }
            }
        }
    }
}
