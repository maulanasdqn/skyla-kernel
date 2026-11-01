use crate::{DType, Gguf, TensorInfo, Value};
use anyhow::{anyhow, bail, ensure, Context as _, Result};
use memmap2::Mmap;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.buf.len())
            .ok_or_else(|| anyhow!("unexpected end of file at {}", self.pos))?;
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

        ensure!(r.bytes(4)? == b"GGUF", "{} is not a GGUF file", path.display());
        let version = r.u32()?;
        ensure!(matches!(version, 2 | 3), "unsupported GGUF version {version}");
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

        let align = metadata.get("general.alignment").and_then(Value::as_u64).unwrap_or(32) as usize;
        let data_start = r.pos.div_ceil(align) * align;

        let mut tensors = BTreeMap::new();
        for (name, dims, dtype, rel) in infos {
            let n: u64 = dims.iter().product();
            let n_bytes = match dtype.block() {
                Ok((per_block, bytes)) => (n as usize).div_ceil(per_block) * bytes,
                Err(_) => 0,
            };
            let offset = data_start + rel;
            ensure!(offset + n_bytes <= mmap.len(), "tensor {name} runs past end of file");
            let info = TensorInfo {
                name: name.clone(),
                dims,
                dtype,
                offset,
                n_bytes,
            };
            tensors.insert(name, info);
        }

        Ok(Self {
            mmap,
            version,
            metadata,
            tensors,
        })
    }
}
