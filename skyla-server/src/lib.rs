mod application;
mod domain;
mod error;
mod infrastructure;

use anyhow::Result;
use application::{EngineHandle, ImageService};
use skyla_engine::KvPrecision;
use std::net::SocketAddr;
use std::path::PathBuf;
use tokio::net::TcpListener;

pub struct Config {
    pub model: PathBuf,
    pub context: usize,
    pub kv: KvPrecision,
    pub addr: SocketAddr,
    pub web: PathBuf,
    pub models: Option<PathBuf>,
    pub image_models: PathBuf,
    pub image_tool: PathBuf,
}

pub fn serve(config: Config) -> Result<()> {
    let root = config.models.clone().unwrap_or_else(|| {
        let dir = config.model.parent().unwrap_or(&config.model);
        dir.parent().unwrap_or(dir).to_path_buf()
    });
    println!("models under {}", root.display());
    println!("loading {} ...", config.model.display());
    let engine = EngineHandle::spawn(config.model.clone(), root, config.context, config.kv)?;
    let images = ImageService::new(config.image_models.clone(), config.image_tool.clone());
    println!("image models under {}", config.image_models.display());
    let runtime = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    runtime.block_on(async move {
        let app = infrastructure::http::router(engine, images, &config.web);
        let listener = TcpListener::bind(config.addr).await?;
        println!("chat UI on http://{}", listener.local_addr()?);
        axum::serve(listener, app).await?;
        Ok(())
    })
}
