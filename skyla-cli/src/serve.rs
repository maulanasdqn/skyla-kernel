use crate::KvCache;
use anyhow::Result;
use clap::Args;
use std::path::PathBuf;

#[derive(Args)]
pub struct ServeArgs {
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
    #[arg(long, help = "Directory with Core ML Stable Diffusion models (default: ~/sd-models)")]
    image_models: Option<PathBuf>,
    #[arg(
        long,
        default_value = "skyla-image/.build/release/skyla-image",
        help = "Image generation helper"
    )]
    image_tool: PathBuf,
}

pub fn run(args: ServeArgs) -> Result<()> {
    let home = PathBuf::from(std::env::var("HOME").unwrap_or_default());
    skyla_server::serve(skyla_server::Config {
        model: args.model,
        context: args.ctx,
        kv: args.kv.precision(),
        addr: args.addr,
        web: args.web,
        models: args.models,
        image_models: args.image_models.unwrap_or_else(|| home.join("sd-models")),
        image_tool: args.image_tool,
    })
}
