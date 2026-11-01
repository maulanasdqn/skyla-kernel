use anyhow::Result;
use skyla_chat::{Piece, Sampler, Session};
use skyla_engine::KvPrecision;
use skyla_gguf::Gguf;
use skyla_tok::Tokenizer;
use std::io::{BufRead, IsTerminal, Write};
use std::path::Path;

pub struct Options {
    pub ctx: usize,
    pub kv: KvPrecision,
    pub max_tokens: usize,
    pub system: Option<String>,
    pub sampler: Sampler,
}

pub fn run(path: &Path, mut opts: Options) -> Result<()> {
    let g = Gguf::open(path)?;
    let tok = Tokenizer::from_gguf(&g)?;
    let mut session = Session::new(&g, &tok, opts.ctx, opts.kv)?;
    let info = session.info();
    println!(
        "{} on {} | context {} tokens | /reset clears the conversation, empty line quits\n",
        info.model, info.device, info.context
    );

    let stdin = std::io::stdin();
    let mut out = std::io::stdout();
    let (dim, reset) = match out.is_terminal() {
        true => ("\x1b[90m", "\x1b[0m"),
        false => ("", ""),
    };
    loop {
        print!("> ");
        out.flush()?;
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 || line.trim().is_empty() {
            return Ok(());
        }
        if line.trim() == "/reset" {
            session.reset();
            println!("conversation cleared\n");
            continue;
        }

        let mut write = |piece: Piece| {
            let text = match piece {
                Piece::ThinkStart => format!("{dim}<think>"),
                Piece::ThinkEnd => format!("</think>{reset}\n"),
                Piece::Text { text, .. } => text,
            };
            out.write_all(text.as_bytes()).and_then(|_| out.flush()).is_ok()
        };
        let system = opts.system.as_deref();
        match session.turn(&line, system, &mut opts.sampler, opts.max_tokens, &mut write) {
            Ok(s) => println!(
                "{reset}\n{dim}[{} prompt tokens in {:.0} ms | {} tokens at {:.2} tok/s | context {}/{}]{reset}\n",
                s.prompt_tokens, s.prefill_ms, s.tokens, s.tokens_per_second, s.position, s.context
            ),
            Err(e) => println!("{reset}\n{e}\n"),
        }
    }
}
