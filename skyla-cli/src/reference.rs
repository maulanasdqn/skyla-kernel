use anyhow::Result;
use skyla_gguf::Gguf;
use skyla_ref::{argmax, softmax, Model};
use std::path::Path;
use std::time::Instant;

fn top_k(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut probs = logits.to_vec();
    softmax(&mut probs);
    let mut top: Vec<(u32, f32)> = probs.iter().enumerate().map(|(i, &p)| (i as u32, p)).collect();
    top.sort_by(|a, b| b.1.total_cmp(&a.1));
    top.truncate(k);
    top
}

pub fn run(path: &Path, prompt: &[u32], n: usize, expect: &[u32]) -> Result<()> {
    let g = Gguf::open(path)?;
    let mut model = Model::load(&g)?;
    println!("{:?}", model.config);

    let t = Instant::now();
    let mut logits = Vec::new();
    for &tok in prompt {
        logits = model.forward(tok);
    }
    println!("prefill {} tok in {:.2?}", prompt.len(), t.elapsed());

    if !expect.is_empty() {
        return check(&mut model, logits, expect);
    }

    println!("top-5 after prompt: {:?}", top_k(&logits, 5));
    let t = Instant::now();
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let next = argmax(&logits);
        out.push(next.to_string());
        logits = model.forward(next);
    }
    let rate = n as f64 / t.elapsed().as_secs_f64();
    println!("generated: {}", out.join(","));
    println!("decode {n} tok at {rate:.2} tok/s");
    Ok(())
}

fn check(model: &mut Model, mut logits: Vec<f32>, expect: &[u32]) -> Result<()> {
    let mut mismatches = 0;
    for (step, &want) in expect.iter().enumerate() {
        let top = top_k(&logits, 2);
        let mut probs = logits.clone();
        softmax(&mut probs);
        let p_want = probs[want as usize];
        let mark = if top[0].0 == want { "ok" } else { "DIFF" };
        if top[0].0 != want {
            mismatches += 1;
        }
        println!(
            "step {step:>3}  want {want:>6} p={p_want:.4}  top1 {:>6} p={:.4}  top2 {:>6} p={:.4}  {mark}",
            top[0].0, top[0].1, top[1].0, top[1].1
        );
        logits = model.forward(want);
    }
    println!("{mismatches} / {} steps disagree with the expected token", expect.len());
    Ok(())
}
