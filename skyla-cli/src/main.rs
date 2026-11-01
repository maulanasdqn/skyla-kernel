mod bench;
mod chat;
mod inspect;
mod probe;
mod reference;
mod run;
mod sample;

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
    #[command(about = "Tokenize text with the model's tokenizer and print the ids")]
    Tokenize { model: PathBuf, text: String },
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
    #[command(about = "Chat with the model, streaming tokens from the GPU engine")]
    Chat {
        model: PathBuf,
        #[arg(long, default_value_t = 8192, help = "Context length (KV cache size)")]
        ctx: usize,
        #[arg(long, default_value_t = 4096, help = "Maximum tokens per reply, thinking included")]
        max_tokens: usize,
        #[arg(long, default_value_t = 0.6, help = "Sampling temperature; 0 is greedy")]
        temp: f32,
        #[arg(long, default_value_t = 40)]
        top_k: usize,
        #[arg(long, default_value_t = 0.95)]
        top_p: f32,
        #[arg(long, help = "Random seed (default: current time)")]
        seed: Option<u64>,
        #[arg(long, help = "System prompt")]
        system: Option<String>,
    },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Probe => probe::run(),
        Command::BenchBw { size_mib, reps } => bench::bandwidth(size_mib << 20, reps),
        Command::GgufInfo { model } => inspect::gguf_info(&model),
        Command::Tokenize { model, text } => inspect::tokenize(&model, &text),
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
        Command::Chat {
            model,
            ctx,
            max_tokens,
            temp,
            top_k,
            top_p,
            seed,
            system,
        } => {
            let seed = seed.unwrap_or_else(|| {
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(1, |d| d.as_nanos() as u64)
            });
            let sampler = sample::Sampler::new(temp, top_k, top_p, seed);
            let opts = chat::Options {
                ctx,
                max_tokens,
                system,
                sampler,
            };
            chat::run(&model, opts)
        }
    }
}
