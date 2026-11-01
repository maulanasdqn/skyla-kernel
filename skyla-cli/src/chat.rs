use crate::sample::Sampler;
use anyhow::Result;
use skyla_engine::Engine;
use skyla_gguf::Gguf;
use skyla_tok::Tokenizer;
use std::io::{BufRead, IsTerminal, Write};
use std::path::Path;
use std::time::Instant;

const DIM: &str = "\x1b[90m";
const RESET: &str = "\x1b[0m";

pub struct Options {
    pub ctx: usize,
    pub kv: skyla_engine::KvPrecision,
    pub max_tokens: usize,
    pub system: Option<String>,
    pub sampler: Sampler,
}

struct Stream {
    pending: Vec<u8>,
}

impl Stream {
    fn push(&mut self, bytes: &[u8], out: &mut impl Write) -> Result<()> {
        self.pending.extend_from_slice(bytes);
        let valid = match std::str::from_utf8(&self.pending) {
            Ok(s) => s.len(),
            Err(e) if e.error_len().is_none() => e.valid_up_to(),
            Err(_) => self.pending.len(),
        };
        let text = String::from_utf8_lossy(&self.pending[..valid]).into_owned();
        self.pending.drain(..valid);
        out.write_all(text.as_bytes())?;
        out.flush()?;
        Ok(())
    }
}

pub fn run(path: &Path, mut opts: Options) -> Result<()> {
    let g = Gguf::open(path)?;
    let tok = Tokenizer::from_gguf(&g)?;
    let mut engine = Engine::new(&g, opts.ctx, opts.kv, false)?;
    let think = (tok.id("<think>"), tok.id("</think>"));
    let bos = String::from_utf8_lossy(&tok.decode(tok.bos)).into_owned();
    println!(
        "{} on {} | context {} tokens | /reset clears the conversation, empty line quits\n",
        g.meta_str("general.name").unwrap_or("model"),
        engine.device_name(),
        opts.ctx
    );

    let stdin = std::io::stdin();
    let mut out = std::io::stdout();
    let (dim, reset) = match out.is_terminal() {
        true => (DIM, RESET),
        false => ("", ""),
    };
    let mut first = true;
    loop {
        print!("> ");
        out.flush()?;
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 || line.trim().is_empty() {
            return Ok(());
        }
        if line.trim() == "/reset" {
            engine.reset();
            first = true;
            println!("conversation cleared\n");
            continue;
        }

        let mut prompt = String::new();
        if first {
            prompt.push_str(&bos);
            prompt.push_str(opts.system.as_deref().unwrap_or(""));
        }
        prompt.push_str(&format!("<｜User｜>{}<｜Assistant｜>", line.trim_end()));
        let ids = tok.encode(&prompt)?;
        if engine.pos as usize + ids.len() + opts.max_tokens > opts.ctx {
            println!("context is full; use /reset\n");
            continue;
        }
        first = false;

        let t = Instant::now();
        let mut logits = engine.prefill(&ids)?.to_vec();
        let prefill = t.elapsed();

        let t = Instant::now();
        let mut stream = Stream { pending: Vec::new() };
        let mut n = 0;
        while n < opts.max_tokens {
            let next = opts.sampler.sample(&logits);
            n += 1;
            if next == tok.eos {
                engine.step(next)?;
                break;
            }
            match Some(next) {
                id if id == think.0 => stream.push(format!("{dim}<think>").as_bytes(), &mut out)?,
                id if id == think.1 => stream.push(format!("</think>{reset}\n").as_bytes(), &mut out)?,
                _ => stream.push(&tok.decode(next), &mut out)?,
            }
            logits = engine.step(next)?.to_vec();
        }
        let rate = n as f64 / t.elapsed().as_secs_f64();
        println!(
            "{reset}\n{dim}[{} prompt tokens in {:.1?} | {n} tokens at {rate:.2} tok/s | context {}/{}]{reset}\n",
            ids.len(),
            prefill,
            engine.pos,
            opts.ctx
        );
    }
}
