use super::{handlers, ws};
use crate::application::EngineHandle;
use axum::routing::{get, post};
use axum::Router;
use std::path::Path;
use tower_http::services::{ServeDir, ServeFile};

pub fn router(engine: EngineHandle, web: &Path) -> Router {
    let site = ServeDir::new(web).fallback(ServeFile::new(web.join("index.html")));
    Router::new()
        .route("/api/info", get(handlers::info))
        .route("/api/reset", post(handlers::reset))
        .route("/api/ws", get(ws::upgrade))
        .fallback_service(site)
        .with_state(engine)
}
