use anyhow::Result;
use skyla_engine::Engine;
use skyla_gguf::Gguf;
use skyla_ref::{argmax, softmax, Model};
use std::path::Path;
use std::time::Instant;

pub fn run(
    path: &Path,
    prompt: &[u32],
    n: usize,
    kv: (usize, skyla_engine::KvPrecision),
    check: bool,
    profile: bool,
) -> Result<()> {
    let g = Gguf::open(path)?;
    let t = Instant::now();
    let mut engine = Engine::new(&g, kv.0, kv.1, profile)?;
    println!(
        "{} | {} MiB of packed weights in {} buffers | loaded in {:.2?}",
        engine.device_name(),
        engine.weight_bytes() >> 20,
        engine.weight_buffers(),
        t.elapsed()
    );
    match check {
        true => compare(&g, &mut engine, prompt, n)?,
        false => generate(&mut engine, prompt, n)?,
    }
    if profile {
        print_profile(&engine, true)?;
        print_profile(&engine, false)?;
    }
    Ok(())
}

fn print_profile(engine: &Engine, prefill: bool) -> Result<()> {
    let marks = engine.last_step_profile(prefill)?;
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
    let what = if prefill { "prefill batch" } else { "decode step" };
    println!("\nlast {what} on the GPU: {:.2} ms", total / 1e6);
    for (label, ns, count) in by_label {
        let pct = 100.0 * ns / total;
        println!("  {label:<16} {:>8.3} ms  {pct:>5.1}%  x{count}", ns / 1e6);
    }
    Ok(())
}

fn generate(engine: &mut Engine, prompt: &[u32], n: usize) -> Result<()> {
    let t = Instant::now();
    let mut next = argmax(engine.prefill(prompt)?);
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
        "prefill {} tok in {:.2?} ({:.1} tok/s) | decode {} tok at {:.2} tok/s",
        prompt.len(),
        prefill,
        prompt.len() as f64 / prefill.as_secs_f64(),
        n - 1,
        (n - 1) as f64 / decode.as_secs_f64()
    );
    Ok(())
}

fn report(i: usize, gpu: &[f32], cpu: &[f32], worst: &mut f32, disagreements: &mut usize) -> u32 {
    let diff = gpu.iter().zip(cpu).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    *worst = worst.max(diff);
    let (g_top, c_top) = (argmax(gpu), argmax(cpu));
    let mut p = cpu.to_vec();
    softmax(&mut p);
    let mark = match g_top == c_top {
        true => "ok",
        false => {
            *disagreements += 1;
            "DIFF"
        }
    };
    println!(
        "pos {i:>3}  gpu {g_top:>6}  cpu {c_top:>6} (p={:.4} vs {:.4})  max|dlogit| {diff:.4}  {mark}",
        p[c_top as usize], p[g_top as usize]
    );
    g_top
}

fn compare(g: &Gguf, engine: &mut Engine, prompt: &[u32], n: usize) -> Result<()> {
    let mut reference = Model::load(g)?;
    let mut worst = 0.0f32;
    let mut disagreements = 0;

    let gpu = engine.prefill(prompt)?.to_vec();
    let mut cpu = Vec::new();
    for &tok in prompt {
        cpu = reference.forward(tok);
    }
    let mut next = report(prompt.len() - 1, &gpu, &cpu, &mut worst, &mut disagreements);

    for i in 1..n {
        let gpu = engine.step(next)?.to_vec();
        let cpu = reference.forward(next);
        next = report(prompt.len() - 1 + i, &gpu, &cpu, &mut worst, &mut disagreements);
    }
    println!(
        "worst |dlogit| {worst:.4}, {disagreements} top-1 disagreements over {n} steps (prompt via batched prefill)"
    );
    Ok(())
}
