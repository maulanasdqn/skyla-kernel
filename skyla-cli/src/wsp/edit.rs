use anyhow::{anyhow, ensure, Result};
use serde_json::Value;
use skyla_gguf::{dequantize, quantize_q8_0, DType, Gguf};
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

struct Change {
    offset: usize,
    bytes: Vec<u8>,
}

fn direction(path: &Path, from: usize) -> Result<Vec<f32>> {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    let entry = v["layers"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|e| e["layer"].as_u64() == Some(from as u64));
    let dir = &entry.ok_or_else(|| anyhow!("the direction file has no layer {from}"))?["dir"];
    let out: Vec<f32> = dir
        .as_array()
        .ok_or_else(|| anyhow!("no direction for layer {from}"))?
        .iter()
        .filter_map(Value::as_f64)
        .map(|x| x as f32)
        .collect();
    Ok(out)
}

fn layer(g: &Gguf, m: usize, r: &[f32], columns: usize) -> Result<Change> {
    let t = g.tensor(&format!("blk.{m}.ffn_down.weight"))?;
    ensure!(
        t.dtype == DType::Q8_0,
        "blk.{m}.ffn_down is {:?}; only Q8_0 can be edited",
        t.dtype
    );
    let (n_ff, n_embd) = (t.dims[0] as usize, t.dims[1] as usize);
    ensure!(r.len() == n_embd, "direction has {} dims, model has {n_embd}", r.len());
    let mut w = vec![0f32; n_ff * n_embd];
    dequantize(t.dtype, g.data(t), &mut w)?;

    let mut s = vec![0f32; n_ff];
    for (row, ro) in w.chunks(n_ff).zip(r) {
        s.iter_mut().zip(row).for_each(|(s, x)| *s += ro * x);
    }
    let mut order: Vec<usize> = (0..n_ff).collect();
    order.sort_by(|a, b| s[*b].abs().total_cmp(&s[*a].abs()));
    let k = if columns == 0 { n_ff } else { columns.min(n_ff) };
    let all: f32 = s.iter().map(|x| x * x).sum();
    let kept: f32 = order[..k].iter().map(|&j| s[j] * s[j]).sum();
    let top: Vec<String> = order[..5].iter().map(|&j| format!("{j}:{:+.3}", s[j])).collect();
    println!(
        "layer {m:>2}  writes {:.3} along the direction, top {k} columns carry {:.0}%  [{}]",
        all.sqrt(),
        100.0 * kept / all.max(1e-12),
        top.join(" ")
    );

    for row in 0..n_embd {
        for &j in &order[..k] {
            w[row * n_ff + j] -= r[row] * s[j];
        }
    }
    let mut bytes = vec![0u8; t.n_bytes];
    quantize_q8_0(&w, &mut bytes);
    let changed = bytes.iter().zip(g.data(t)).filter(|(a, b)| a != b).count();
    println!("          {changed} of {} bytes change", t.n_bytes);
    Ok(Change {
        offset: t.offset,
        bytes,
    })
}

pub fn run(model: &Path, dir: &Path, from: usize, layers: &[usize], columns: usize, out: Option<&Path>) -> Result<()> {
    let g = Gguf::open(model)?;
    let r = direction(dir, from)?;
    let n_layer = g.meta_u64(&format!("{}.block_count", g.meta_str("general.architecture")?))? as usize;
    let targets: Vec<usize> = if layers.is_empty() {
        (0..n_layer).collect()
    } else {
        layers.to_vec()
    };
    let changes = targets
        .iter()
        .map(|&m| layer(&g, m, &r, columns))
        .collect::<Result<Vec<_>>>()?;
    let Some(out) = out else { return Ok(()) };
    ensure!(out != model, "write the edit to a new file");
    std::fs::copy(model, out)?;
    let mut file = std::fs::OpenOptions::new().write(true).open(out)?;
    for c in changes {
        file.seek(SeekFrom::Start(c.offset as u64))?;
        file.write_all(&c.bytes)?;
    }
    println!("wrote {}", out.display());
    Ok(())
}
