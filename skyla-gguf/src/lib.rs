mod quant;
mod reader;

pub use quant::dequantize;

use anyhow::{anyhow, bail, ensure, Result};
use memmap2::Mmap;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    Q8_0,
    Q4K,
    Q6K,
    Other(u32),
}

impl DType {
    fn from_raw(v: u32) -> Self {
        match v {
            0 => Self::F32,
            1 => Self::F16,
            8 => Self::Q8_0,
            12 => Self::Q4K,
            14 => Self::Q6K,
            v => Self::Other(v),
        }
    }

    pub fn block(self) -> Result<(usize, usize)> {
        Ok(match self {
            Self::F32 => (1, 4),
            Self::F16 => (1, 2),
            Self::Q8_0 => (32, 34),
            Self::Q4K => (256, 144),
            Self::Q6K => (256, 210),
            Self::Other(v) => bail!("unsupported ggml type {v}"),
        })
    }

    pub fn row_bytes(self, n: usize) -> Result<usize> {
        let (per_block, bytes) = self.block()?;
        ensure!(
            n.is_multiple_of(per_block),
            "row of {n} elements is not a multiple of {per_block}"
        );
        Ok(n / per_block * bytes)
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

impl Gguf {
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
        self.meta(key)?.as_f64().ok_or_else(|| anyhow!("{key} is not a number"))
    }

    pub fn meta_str(&self, key: &str) -> Result<&str> {
        self.meta(key)?.as_str().ok_or_else(|| anyhow!("{key} is not a string"))
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors.get(name).ok_or_else(|| anyhow!("missing tensor {name}"))
    }

    pub fn data(&self, t: &TensorInfo) -> &[u8] {
        &self.mmap[t.offset..t.offset + t.n_bytes]
    }

    pub fn to_f32(&self, name: &str) -> Result<Vec<f32>> {
        let t = self.tensor(name)?;
        let mut out = vec![0.0; t.n_elements() as usize];
        dequantize(t.dtype, self.data(t), &mut out)?;
        Ok(out)
    }
}
