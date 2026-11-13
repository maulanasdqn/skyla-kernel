mod direction;
mod edit;
mod math;

use anyhow::Result;
use clap::Subcommand;
use std::path::PathBuf;

#[derive(Subcommand)]
pub enum Cmd {
    #[command(about = "Solve GSM8K problems on the GPU and record every generation")]
    Math {
        model: PathBuf,
        #[arg(long, help = "GSM8K jsonl with question and answer fields")]
        data: PathBuf,
        #[arg(long, default_value_t = 0)]
        skip: usize,
        #[arg(short, default_value_t = 32)]
        n: usize,
        #[arg(long, default_value_t = 4096)]
        max_tokens: usize,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        #[arg(long)]
        out: PathBuf,
    },
    #[command(about = "Find the residual direction that separates repeating tokens from fresh ones")]
    Direction {
        model: PathBuf,
        #[arg(long, help = "Runs written by wsp math")]
        runs: PathBuf,
        #[arg(
            long,
            default_value_t = 16,
            help = "A token repeats when the n-gram ending at it occurred before"
        )]
        ngram: usize,
        #[arg(long, value_delimiter = ',', help = "Residual layers to measure (default: all)")]
        layers: Vec<usize>,
        #[arg(long)]
        out: PathBuf,
    },
    #[command(about = "Remove a direction from the ffn_down columns that write it most and save a new GGUF")]
    Edit {
        model: PathBuf,
        #[arg(long)]
        direction: PathBuf,
        #[arg(long, help = "Residual layer the direction was measured after")]
        from: usize,
        #[arg(long, value_delimiter = ',', help = "Layers whose ffn_down is edited")]
        layers: Vec<usize>,
        #[arg(long, default_value_t = 64, help = "Columns edited per layer (0 = all)")]
        columns: usize,
        #[arg(long, help = "Print what would change without writing")]
        dry_run: bool,
        #[arg(long)]
        out: Option<PathBuf>,
    },
}

pub fn run(cmd: Cmd) -> Result<()> {
    match cmd {
        Cmd::Math {
            model,
            data,
            skip,
            n,
            max_tokens,
            seed,
            out,
        } => math::run(&model, &data, skip..skip + n, max_tokens, seed, &out),
        Cmd::Direction {
            model,
            runs,
            ngram,
            layers,
            out,
        } => direction::run(&model, &runs, ngram, &layers, &out),
        Cmd::Edit {
            model,
            direction,
            from,
            layers,
            columns,
            dry_run,
            out,
        } => edit::run(
            &model,
            &direction,
            from,
            &layers,
            columns,
            out.filter(|_| !dry_run).as_deref(),
        ),
    }
}
