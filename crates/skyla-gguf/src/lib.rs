//! Memory-mapped reader for GGUF v2/v3 checkpoints.
//!
//! Tensor dimensions follow ggml order: `dims[0]` is the contiguous (input) dimension,
//! so a matrix with `dims = [n_in, n_out]` stores one output row per `n_in` elements.

use anyhow::{anyhow, bail, ensure, Context as _, Result};
use half::f16;
use memmap2::Mmap;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    Q8_0,
    Other(u32),
}

impl DType {
    fn from_raw(v: u32) -> Self {
        match v {
            0 => Self::F32,
            1 => Self::F16,
            8 => Self::Q8_0,
            v => Self::Other(v),
        }
    }

    /// (elements per block, bytes per block)
    pub fn block(self) -> Result<(usize, usize)> {
        Ok(match self {
            Self::F32 => (1, 4),
            Self::F16 => (1, 2),
            Self::Q8_0 => (32, 34),
            Self::Other(v) => bail!("unsupported ggml type {v}"),
        })
    }
}

#[derive(Debug, Clone)]
pub enum Value {
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    Str(String),
    Array(Vec<Value>),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Self::U64(v) => Some(v),
            Self::I64(v) => u64::try_from(v).ok(),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Self::F64(v) => Some(v),
            Self::U64(v) => Some(v as f64),
            Self::I64(v) => Some(v as f64),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Self::Array(a) => Some(a),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub dtype: DType,
    /// Absolute byte offset in the file.
    pub offset: usize,
    pub n_bytes: usize,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product()
    }
}

pub struct Gguf {
    mmap: Mmap,
    pub version: u32,
    pub metadata: BTreeMap<String, Value>,
    pub tensors: BTreeMap<String, TensorInfo>,
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.buf.len());
        let end = end.ok_or_else(|| anyhow!("unexpected end of file at {}", self.pos))?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn arr<const N: usize>(&mut self) -> Result<[u8; N]> {
        Ok(self.bytes(N)?.try_into().unwrap())
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.arr()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.arr()?))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.bytes(n)?).into_owned())
    }
    fn value(&mut self, ty: u32) -> Result<Value> {
        Ok(match ty {
            0 => Value::U64(self.arr::<1>()?[0] as u64),
            1 => Value::I64(self.arr::<1>()?[0] as i8 as i64),
            2 => Value::U64(u16::from_le_bytes(self.arr()?) as u64),
            3 => Value::I64(i16::from_le_bytes(self.arr()?) as i64),
            4 => Value::U64(self.u32()? as u64),
            5 => Value::I64(i32::from_le_bytes(self.arr()?) as i64),
            6 => Value::F64(f32::from_le_bytes(self.arr()?) as f64),
            7 => Value::Bool(self.arr::<1>()?[0] != 0),
            8 => Value::Str(self.string()?),
            9 => {
                let inner = self.u32()?;
                let n = self.u64()? as usize;
                Value::Array((0..n).map(|_| self.value(inner)).collect::<Result<_>>()?)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(i64::from_le_bytes(self.arr()?)),
            12 => Value::F64(f64::from_le_bytes(self.arr()?)),
            t => bail!("unknown metadata value type {t}"),
        })
    }
}

impl Gguf {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let mmap = unsafe { Mmap::map(&file) }?;
        let mut r = Reader { buf: &mmap, pos: 0 };

        ensure!(
            r.bytes(4)? == b"GGUF",
            "{} is not a GGUF file",
            path.display()
        );
        let version = r.u32()?;
        ensure!(
            matches!(version, 2 | 3),
            "unsupported GGUF version {version}"
        );
        let n_tensors = r.u64()?;
        let n_kv = r.u64()?;

        let mut metadata = BTreeMap::new();
        for _ in 0..n_kv {
            let key = r.string()?;
            let ty = r.u32()?;
            metadata.insert(key, r.value(ty)?);
        }

        let mut infos = Vec::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name = r.string()?;
            let n_dims = r.u32()? as usize;
            let dims = (0..n_dims).map(|_| r.u64()).collect::<Result<Vec<_>>>()?;
            let dtype = DType::from_raw(r.u32()?);
            let rel = r.u64()? as usize;
            infos.push((name, dims, dtype, rel));
        }

        let align = metadata
            .get("general.alignment")
            .and_then(Value::as_u64)
            .unwrap_or(32) as usize;
        let data_start = r.pos.div_ceil(align) * align;

        let mut tensors = BTreeMap::new();
        for (name, dims, dtype, rel) in infos {
            let n: u64 = dims.iter().product();
            // Unsupported types still get listed; their size is only needed when read.
            let n_bytes = match dtype.block() {
                Ok((per_block, bytes)) => (n as usize).div_ceil(per_block) * bytes,
                Err(_) => 0,
            };
            let offset = data_start + rel;
            ensure!(
                offset + n_bytes <= mmap.len(),
                "tensor {name} runs past end of file"
            );
            tensors.insert(
                name.clone(),
                TensorInfo {
                    name,
                    dims,
                    dtype,
                    offset,
                    n_bytes,
                },
            );
        }

        Ok(Self {
            mmap,
            version,
            metadata,
            tensors,
        })
    }

    pub fn meta(&self, key: &str) -> Result<&Value> {
        self.metadata
            .get(key)
            .ok_or_else(|| anyhow!("missing metadata key {key}"))
    }

    pub fn meta_u64(&self, key: &str) -> Result<u64> {
        self.meta(key)?
            .as_u64()
            .ok_or_else(|| anyhow!("{key} is not an integer"))
    }

    pub fn meta_f64(&self, key: &str) -> Result<f64> {
        self.meta(key)?
            .as_f64()
            .ok_or_else(|| anyhow!("{key} is not a number"))
    }

    pub fn meta_str(&self, key: &str) -> Result<&str> {
        self.meta(key)?
            .as_str()
            .ok_or_else(|| anyhow!("{key} is not a string"))
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors
            .get(name)
            .ok_or_else(|| anyhow!("missing tensor {name}"))
    }

    /// Raw bytes of a tensor, exactly as stored in the file.
    pub fn data(&self, t: &TensorInfo) -> &[u8] {
        &self.mmap[t.offset..t.offset + t.n_bytes]
    }

    /// Dequantizes a whole tensor to f32 (for small tensors such as norms).
    pub fn to_f32(&self, name: &str) -> Result<Vec<f32>> {
        let t = self.tensor(name)?;
        let mut out = vec![0.0; t.n_elements() as usize];
        dequantize(t.dtype, self.data(t), &mut out)?;
        Ok(out)
    }
}

/// Dequantizes `out.len()` elements from `src`.
pub fn dequantize(dtype: DType, src: &[u8], out: &mut [f32]) -> Result<()> {
    match dtype {
        DType::F32 => {
            for (o, b) in out.iter_mut().zip(src.as_chunks::<4>().0) {
                *o = f32::from_le_bytes(*b);
            }
        }
        DType::F16 => {
            for (o, b) in out.iter_mut().zip(src.as_chunks::<2>().0) {
                *o = f16::from_le_bytes(*b).to_f32();
            }
        }
        DType::Q8_0 => {
            for (o, b) in out.chunks_mut(32).zip(src.as_chunks::<34>().0) {
                let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
                for (x, &q) in o.iter_mut().zip(&b[2..]) {
                    *x = d * (q as i8) as f32;
                }
            }
        }
        DType::Other(v) => bail!("cannot dequantize ggml type {v}"),
    }
    Ok(())
}
