use anyhow::Result;
use skyla_engine::Engine;
use skyla_gguf::Gguf;
use skyla_ref::{argmax, softmax, Model};
use std::path::Path;
use std::time::Instant;

pub fn run(path: &Path, prompt: &[u32], n: usize, max_ctx: usize, check: bool, profile: bool) -> Result<()> {
    let g = Gguf::open(path)?;
    let t = Instant::now();
    let mut engine = Engine::new(&g, max_ctx, profile)?;
    println!(
        "{} | {} MiB of packed weights | loaded in {:.2?}",
        engine.device_name(),
        engine.weight_bytes() >> 20,
        t.elapsed()
    );
    match check {
        true => compare(&g, &mut engine, prompt, n)?,
        false => generate(&mut engine, prompt, n)?,
    }
    if profile {
        print_profile(&engine)?;
    }
    Ok(())
}

fn print_profile(engine: &Engine) -> Result<()> {
    let marks = engine.last_step_profile()?;
    let total: f64 = marks.iter().map(|m| m.1).sum();
    let mut by_label: Vec<(&str, f64, usize)> = Vec::new();
    for (label, ns) in marks {
        match by_label.iter_mut().find(|e| e.0 == label) {
            Some(e) => {
                e.1 += ns;
                e.2 += 1;
            }
            None => by_label.push((label, ns, 1)),
        }
    }
    by_label.sort_by(|a, b| b.1.total_cmp(&a.1));
    println!("\nlast decode step on the GPU: {:.2} ms", total / 1e6);
    for (label, ns, count) in by_label {
        let pct = 100.0 * ns / total;
        println!("  {label:<16} {:>8.3} ms  {pct:>5.1}%  x{count}", ns / 1e6);
    }
    Ok(())
}

fn generate(engine: &mut Engine, prompt: &[u32], n: usize) -> Result<()> {
    let t = Instant::now();
    let mut next = 0;
    for &tok in prompt {
        next = argmax(engine.step(tok)?);
    }
    let prefill = t.elapsed();

    let t = Instant::now();
    let mut out = vec![next.to_string()];
    for _ in 1..n {
        next = argmax(engine.step(next)?);
        out.push(next.to_string());
    }
    let decode = t.elapsed();
    println!("generated: {}", out.join(","));
    println!(
        "prefill {} tok in {:.2?} | decode {} tok at {:.2} tok/s",
        prompt.len(),
        prefill,
        n - 1,
        (n - 1) as f64 / decode.as_secs_f64()
    );
    Ok(())
}

fn compare(g: &Gguf, engine: &mut Engine, prompt: &[u32], n: usize) -> Result<()> {
    let mut reference = Model::load(g)?;
    let mut tokens = prompt.to_vec();
    let mut worst = 0.0f32;
    let mut disagreements = 0;
    for i in 0..prompt.len() + n - 1 {
        let gpu = engine.step(tokens[i])?.to_vec();
        let cpu = reference.forward(tokens[i]);
        let diff = gpu.iter().zip(&cpu).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        worst = worst.max(diff);
        let (g_top, c_top) = (argmax(&gpu), argmax(&cpu));
        if i + 1 < prompt.len() {
            continue;
        }
        let mut p = cpu.clone();
        softmax(&mut p);
        let mark = match g_top == c_top {
            true => "ok",
            false => {
                disagreements += 1;
                "DIFF"
            }
        };
        println!(
            "pos {i:>3}  gpu {g_top:>6}  cpu {c_top:>6} (p={:.4} vs {:.4})  max|dlogit| {diff:.4}  {mark}",
            p[c_top as usize], p[g_top as usize]
        );
        tokens.push(g_top);
    }
    println!("worst |dlogit| {worst:.4}, {disagreements} top-1 disagreements over {n} steps");
    Ok(())
}
