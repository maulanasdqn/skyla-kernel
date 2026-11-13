use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use skyla_chat::Sampler;
use skyla_engine::{Engine, KvPrecision};
use skyla_gguf::Gguf;
use skyla_tok::Tokenizer;
use std::io::{BufRead, BufReader, Write};
use std::ops::Range;
use std::path::Path;
use std::time::Instant;

const SUFFIX: &str = "\n\nPlease reason step by step, and put your final answer within \\boxed{}.";

pub fn problems(path: &Path, range: Range<usize>) -> Result<Vec<(String, String)>> {
    let file = BufReader::new(std::fs::File::open(path)?);
    file.lines()
        .skip(range.start)
        .take(range.len())
        .map(|line| {
            let v: Value = serde_json::from_str(&line?)?;
            let answer = v["answer"].as_str().unwrap_or_default();
            let gold = answer.rsplit("####").next().unwrap_or_default().trim().to_string();
            Ok((v["question"].as_str().unwrap_or_default().to_string(), gold))
        })
        .collect()
}

fn number(s: &str) -> Option<f64> {
    let clean: String = s
        .chars()
        .filter(|c| c.is_ascii_digit() || matches!(c, '.' | '-'))
        .collect();
    clean.trim_end_matches('.').parse().ok()
}

pub fn boxed(text: &str) -> Option<String> {
    let start = text.rfind("\\boxed{")? + 7;
    let mut depth = 1;
    for (i, c) in text[start..].char_indices() {
        depth += match c {
            '{' => 1,
            '}' => -1,
            _ => 0,
        };
        if depth == 0 {
            return Some(text[start..start + i].to_string());
        }
    }
    None
}

pub fn correct(text: &str, gold: &str) -> bool {
    match (boxed(text).as_deref().and_then(number), number(gold)) {
        (Some(a), Some(b)) => (a - b).abs() < 1e-6,
        _ => false,
    }
}

pub fn run(model: &Path, data: &Path, range: Range<usize>, max_tokens: usize, seed: u64, out: &Path) -> Result<()> {
    let g = Gguf::open(model)?;
    let tok = Tokenizer::from_gguf(&g)?;
    let stop = tok.id("<|im_end|>").ok_or_else(|| anyhow!("no <|im_end|> token"))?;
    let mut engine = Engine::new(&g, max_tokens + 1024, KvPrecision::F16KeysQ8Values, false)?;
    let mut file = std::fs::File::create(out)?;
    let (mut solved, mut limited, mut total) = (0, 0, 0);
    let started = Instant::now();
    for (i, (question, gold)) in problems(data, range.clone())?.into_iter().enumerate() {
        let prompt = format!("<|im_start|>user\n{question}{SUFFIX}<|im_end|>\n<|im_start|>assistant\n");
        let ids = tok.encode(&prompt)?;
        let mut sampler = Sampler::new(0.6, 20, 0.95, seed.wrapping_add((range.start + i) as u64)).without_penalty();
        engine.reset();
        let mut logits = engine.prefill(&ids)?.to_vec();
        let mut gen = Vec::new();
        let mut hit_limit = true;
        while gen.len() < max_tokens {
            let next = sampler.sample(&logits);
            if next == stop {
                hit_limit = false;
                break;
            }
            gen.push(next);
            logits = engine.step(next)?.to_vec();
        }
        let bytes: Vec<u8> = gen.iter().flat_map(|&t| tok.decode(t)).collect();
        let text = String::from_utf8_lossy(&bytes).into_owned();
        let ok = !hit_limit && correct(&text, &gold);
        total += 1;
        solved += ok as usize;
        limited += hit_limit as usize;
        println!(
            "#{:<4} {} {:>5} tok{}  ({solved}/{total} solved, {limited} hit the limit, {:.0}s)",
            range.start + i,
            if ok { "✓" } else { "✗" },
            gen.len(),
            if hit_limit { "  LIMIT" } else { "" },
            started.elapsed().as_secs_f64()
        );
        let record = json!({"question": question, "gold": gold, "prompt": ids, "ids": gen,
                            "hit_limit": hit_limit, "correct": ok, "text": text});
        writeln!(file, "{record}")?;
    }
    println!(
        "solved {solved}/{total} ({:.1}%), hit the {max_tokens}-token limit {limited}/{total} ({:.1}%)",
        100.0 * solved as f64 / total.max(1) as f64,
        100.0 * limited as f64 / total.max(1) as f64
    );
    Ok(())
}
