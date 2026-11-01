mod bench;
mod chat;
mod inspect;
mod probe;
mod reference;
mod run;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum KvCache {
    Q8,
    F16,
    V8,
}

impl KvCache {
    fn precision(self) -> skyla_engine::KvPrecision {
        match self {
            Self::Q8 => skyla_engine::KvPrecision::Q8,
            Self::F16 => skyla_engine::KvPrecision::F16,
            Self::V8 => skyla_engine::KvPrecision::F16KeysQ8Values,
        }
    }
}

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
        #[arg(long, value_enum, default_value_t = KvCache::V8, help = "KV cache precision (v8: f16 keys, q8 values)")]
        kv: KvCache,
        #[arg(long, help = "Run the CPU reference in lockstep and compare logits every step")]
        check: bool,
        #[arg(long, help = "Time every stage of the last decode step with GPU timestamps")]
        profile: bool,
    },
    #[command(about = "Chat with the model, streaming tokens from the GPU engine")]
    Chat {
        model: PathBuf,
        #[arg(long, default_value_t = 16384, help = "Context length (KV cache size)")]
        ctx: usize,
        #[arg(long, value_enum, default_value_t = KvCache::V8, help = "KV cache precision (v8: f16 keys, q8 values)")]
        kv: KvCache,
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
    #[command(about = "Serve the browser chat UI and stream replies over WebSocket")]
    Serve {
        model: PathBuf,
        #[arg(long, default_value_t = 16384, help = "Context length (KV cache size)")]
        ctx: usize,
        #[arg(long, value_enum, default_value_t = KvCache::V8, help = "KV cache precision")]
        kv: KvCache,
        #[arg(long, default_value = "127.0.0.1:8080", help = "Address to listen on")]
        addr: std::net::SocketAddr,
        #[arg(long, default_value = "skyla-web/dist", help = "Directory with the built web UI")]
        web: PathBuf,
        #[arg(long, help = "Directory scanned for GGUF models (default: two levels above MODEL)")]
        models: Option<PathBuf>,
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
            kv,
            check,
            profile,
        } => run::run(&model, &tokens, n, (ctx, kv.precision()), check, profile),
        Command::Chat {
            model,
            ctx,
            kv,
            max_tokens,
            temp,
            top_k,
            top_p,
            seed,
            system,
        } => {
            let seed = seed.unwrap_or_else(skyla_chat::seed_from_clock);
            let sampler = skyla_chat::Sampler::new(temp, top_k, top_p, seed);
            let opts = chat::Options {
                ctx,
                kv: kv.precision(),
                max_tokens,
                system,
                sampler,
            };
            chat::run(&model, opts)
        }
        Command::Serve {
            model,
            ctx,
            kv,
            addr,
            web,
            models,
        } => skyla_server::serve(skyla_server::Config {
            model,
            context: ctx,
            kv: kv.precision(),
            addr,
            web,
            models,
        }),
    }
}
