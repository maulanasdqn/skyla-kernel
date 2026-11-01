mod application;
mod domain;
mod error;
mod infrastructure;

use anyhow::Result;
use application::EngineHandle;
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
}

pub fn serve(config: Config) -> Result<()> {
    println!("loading {} ...", config.model.display());
    let engine = EngineHandle::spawn(config.model.clone(), config.context, config.kv)?;
    let runtime = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    runtime.block_on(async move {
        let app = infrastructure::http::router(engine, &config.web);
        let listener = TcpListener::bind(config.addr).await?;
        println!("chat UI on http://{}", listener.local_addr()?);
        axum::serve(listener, app).await?;
        Ok(())
    })
}
