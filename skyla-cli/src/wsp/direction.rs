use anyhow::Result;
use serde_json::{json, Value};
use skyla_engine::{Engine, KvPrecision};
use skyla_gguf::Gguf;
use std::collections::HashSet;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

const SCORE_RUNS: usize = 6;

struct Run {
    tokens: Vec<u32>,
    prompt: usize,
    repeat: Vec<bool>,
}

fn ids(v: &Value) -> Vec<u32> {
    v.as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_u64)
        .map(|x| x as u32)
        .collect()
}

fn label(gen: &[u32], n: usize) -> Vec<bool> {
    let mut seen = HashSet::new();
    (0..gen.len())
        .map(|i| i + 1 >= n && !seen.insert(&gen[i + 1 - n..=i]))
        .collect()
}

fn load(path: &Path, ngram: usize) -> Result<Vec<Run>> {
    BufReader::new(std::fs::File::open(path)?)
        .lines()
        .map(|line| {
            let v: Value = serde_json::from_str(&line?)?;
            let (prompt, gen) = (ids(&v["prompt"]), ids(&v["ids"]));
            let repeat = label(&gen, ngram);
            Ok(Run {
                prompt: prompt.len(),
                tokens: [prompt, gen].concat(),
                repeat,
            })
        })
        .collect()
}

fn pass(engine: &mut Engine, runs: &[Run], layers: &[usize], mut f: impl FnMut(usize, bool, &[f32])) -> Result<()> {
    for run in runs {
        engine.reset();
        engine.residuals(&run.tokens, layers, |t, l, x| {
            if t >= run.prompt {
                f(l, run.repeat[t - run.prompt], x);
            }
        })?;
    }
    Ok(())
}

pub fn run(model: &Path, runs: &Path, ngram: usize, layers: &[usize], out: &Path) -> Result<()> {
    let runs = load(runs, ngram)?;
    let g = Gguf::open(model)?;
    let longest = runs.iter().map(|r| r.tokens.len()).max().unwrap_or(1);
    let mut engine = Engine::new(&g, longest + 256, KvPrecision::F16KeysQ8Values, false)?;
    let n = engine.config.n_embd;
    let layers: Vec<usize> = match layers.is_empty() {
        true => (0..engine.config.n_layer).collect(),
        false => layers.to_vec(),
    };
    let n_layer = layers.len();
    let started = Instant::now();

    let mut sums = vec![[vec![0f64; n], vec![0f64; n]]; n_layer];
    let mut counts = vec![[0usize; 2]; n_layer];
    let mut norms = vec![0f64; n_layer];
    pass(&mut engine, &runs, &layers, |l, rep, x| {
        let c = rep as usize;
        sums[l][c].iter_mut().zip(x).for_each(|(s, v)| *s += *v as f64);
        counts[l][c] += 1;
        norms[l] += x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
    })?;
    let total: usize = counts[0].iter().sum();
    println!(
        "{} runs, {} generated tokens ({} repeating), first pass {:.0}s",
        runs.len(),
        total,
        counts[0][1],
        started.elapsed().as_secs_f64()
    );

    let dirs: Vec<Vec<f64>> = (0..n_layer)
        .map(|l| {
            let [f, r] = &sums[l];
            let [nf, nr] = counts[l].map(|c| c.max(1) as f64);
            r.iter().zip(f).map(|(r, f)| r / nr - f / nf).collect()
        })
        .collect();
    let units: Vec<Vec<f32>> = dirs
        .iter()
        .map(|d| {
            let len = d.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
            d.iter().map(|v| (v / len) as f32).collect()
        })
        .collect();

    let mut proj = vec![[Vec::new(), Vec::new()]; n_layer];
    pass(
        &mut engine,
        &runs[..runs.len().min(SCORE_RUNS)],
        &layers,
        |l, rep, x| {
            proj[l][rep as usize].push(x.iter().zip(&units[l]).map(|(a, b)| a * b).sum::<f32>() as f64);
        },
    )?;

    let mut report = Vec::new();
    println!("layer  |diff|/|x|   separation (cohen d)");
    for l in 0..n_layer {
        let stats = proj[l].clone().map(|p| {
            let m = p.iter().sum::<f64>() / p.len().max(1) as f64;
            let v = p.iter().map(|x| (x - m).powi(2)).sum::<f64>() / p.len().max(1) as f64;
            (m, v)
        });
        let d = (stats[1].0 - stats[0].0) / ((stats[0].1 + stats[1].1) / 2.0).sqrt().max(1e-12);
        let len = dirs[l].iter().map(|v| v * v).sum::<f64>().sqrt();
        let rel = len / (norms[l] / total.max(1) as f64);
        println!("{:>5}  {rel:>9.3}   {d:>7.2}", layers[l]);
        report.push(json!({"layer": layers[l], "rel": rel, "cohen_d": d, "dir": units[l]}));
    }
    std::fs::write(
        out,
        serde_json::to_string(&json!({"n_embd": n, "ngram": ngram, "layers": report}))?,
    )?;
    Ok(())
}
