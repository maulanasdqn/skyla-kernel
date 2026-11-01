use anyhow::{ensure, Result};
use skyla_gguf::{dequantize, DType, Gguf};

#[derive(Clone, Copy)]
pub struct Matrix<'a> {
    pub dtype: DType,
    pub data: &'a [u8],
    pub n_in: usize,
    pub n_out: usize,
    row_bytes: usize,
}

impl<'a> Matrix<'a> {
    pub fn new(g: &'a Gguf, name: &str) -> Result<Self> {
        let t = g.tensor(name)?;
        ensure!(
            matches!(t.dims.len(), 2 | 3),
            "{name} is not a matrix or a stack of experts"
        );
        let (n_in, n_out) = (t.dims[0] as usize, t.dims[1] as usize);
        Ok(Self {
            dtype: t.dtype,
            data: g.data(t),
            n_in,
            n_out,
            row_bytes: t.dtype.row_bytes(n_in)?,
        })
    }

    pub fn expert(&self, e: usize) -> Self {
        let size = self.n_out * self.row_bytes;
        Self {
            data: &self.data[e * size..(e + 1) * size],
            ..*self
        }
    }

    pub fn row(&self, r: usize, out: &mut [f32]) {
        let bytes = &self.data[r * self.row_bytes..(r + 1) * self.row_bytes];
        dequantize(self.dtype, bytes, out).expect("dtype checked at load");
    }

    pub fn matvec(&self, x: &[f32], y: &mut [f32]) {
        assert_eq!((x.len(), y.len()), (self.n_in, self.n_out));
        let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
        let per = self.n_out.div_ceil(threads);
        std::thread::scope(|s| {
            for (t, ys) in y.chunks_mut(per).enumerate() {
                s.spawn(move || {
                    let mut row = vec![0.0f32; self.n_in];
                    for (i, yi) in ys.iter_mut().enumerate() {
                        self.row(t * per + i, &mut row);
                        *yi = row.iter().zip(x).map(|(a, b)| a * b).sum();
                    }
                });
            }
        });
    }
}
