use anyhow::Result;
use skyla_gguf::{Gguf, Value};
use std::collections::BTreeMap;
use std::path::Path;

const SAMPLE_TENSORS: [&str; 4] = [
    "token_embd.weight",
    "blk.0.attn_q.weight",
    "blk.0.ffn_down.weight",
    "output.weight",
];

pub fn gguf_info(path: &Path) -> Result<()> {
    let g = Gguf::open(path)?;
    println!("GGUF v{}, {} tensors\n", g.version, g.tensors.len());
    for (k, v) in &g.metadata {
        let shown = match v {
            Value::Array(a) => format!("[{} items]", a.len()),
            Value::Str(s) if s.len() > 80 => format!("{:?}...", &s[..s.floor_char_boundary(80)]),
            v => format!("{v:?}"),
        };
        println!("{k:<48} {shown}");
    }

    let mut by_type: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for t in g.tensors.values() {
        let e = by_type.entry(format!("{:?}", t.dtype)).or_default();
        e.0 += 1;
        e.1 += t.n_bytes;
    }
    println!();
    for (ty, (count, bytes)) in by_type {
        let mib = bytes as f64 / (1 << 20) as f64;
        println!("{ty:<10} {count:>4} tensors  {mib:>8.1} MiB");
    }
    for name in SAMPLE_TENSORS {
        if let Some(t) = g.tensors.get(name) {
            println!("{name:<28} {:?} {:?}", t.dtype, t.dims);
        }
    }
    Ok(())
}
