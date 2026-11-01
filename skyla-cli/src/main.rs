mod bench;
mod inspect;
mod probe;
mod reference;
mod run;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "skyla", version, about = "Ahead-of-time LLM kernels for one AMD Vega iGPU")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    #[command(about = "Print the selected Vulkan device, its features, limits and memory types")]
    Probe,
    #[command(about = "Measure streaming-read bandwidth of every GPU memory type and of the CPU")]
    BenchBw {
        #[arg(long, default_value_t = 512, help = "Bytes read per measurement, in MiB")]
        size_mib: u64,
        #[arg(
            long,
            default_value_t = 7,
            help = "Timed repetitions per configuration (median is reported)"
        )]
        reps: u32,
    },
    #[command(about = "Print a GGUF file's metadata and tensor summary")]
    GgufInfo { model: PathBuf },
    #[command(about = "Greedy-decode with the CPU reference model and print the generated token ids")]
    Ref {
        model: PathBuf,
        #[arg(
            long,
            value_delimiter = ',',
            required = true,
            help = "Prompt as comma-separated token ids (e.g. from llama-tokenize --ids)"
        )]
        tokens: Vec<u32>,
        #[arg(short, default_value_t = 16, help = "Tokens to generate")]
        n: usize,
        #[arg(
            long,
            value_delimiter = ',',
            help = "Teacher-force these continuation ids and report where our top-1 disagrees"
        )]
        expect: Vec<u32>,
    },
    #[command(about = "Greedy-decode on the GPU engine")]
    Run {
        model: PathBuf,
        #[arg(
            long,
            value_delimiter = ',',
            required = true,
            help = "Prompt as comma-separated token ids"
        )]
        tokens: Vec<u32>,
        #[arg(short, default_value_t = 32, help = "Tokens to generate")]
        n: usize,
        #[arg(long, default_value_t = 2048, help = "Maximum context length (KV cache size)")]
        ctx: usize,
        #[arg(long, help = "Run the CPU reference in lockstep and compare logits every step")]
        check: bool,
        #[arg(long, help = "Time every stage of the last decode step with GPU timestamps")]
        profile: bool,
    },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Probe => probe::run(),
        Command::BenchBw { size_mib, reps } => bench::bandwidth(size_mib << 20, reps),
        Command::GgufInfo { model } => inspect::gguf_info(&model),
        Command::Ref {
            model,
            tokens,
            n,
            expect,
        } => reference::run(&model, &tokens, n, &expect),
        Command::Run {
            model,
            tokens,
            n,
            ctx,
            check,
            profile,
        } => run::run(&model, &tokens, n, ctx, check, profile),
    }
}
