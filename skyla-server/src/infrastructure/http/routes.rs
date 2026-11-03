use super::{handlers, images, ws};
use crate::application::{EngineHandle, ImageService};
use axum::routing::{get, post};
use axum::{Extension, Router};
use std::path::Path;
use tower_http::services::{ServeDir, ServeFile};

pub fn router(engine: EngineHandle, images: ImageService, web: &Path) -> Router {
    let site = ServeDir::new(web).fallback(ServeFile::new(web.join("index.html")));
    Router::new()
        .route("/api/info", get(handlers::info))
        .route("/api/models", get(handlers::models))
        .route("/api/reset", post(handlers::reset))
        .route("/api/ws", get(ws::upgrade))
        .route("/api/images", get(images::gallery))
        .route("/api/images/models", get(images::models))
        .route("/api/images/status", get(images::status))
        .route("/api/images/generate", post(images::generate))
        .route("/api/images/cancel", post(images::cancel))
        .nest_service("/images", ServeDir::new(images.outputs()))
        .fallback_service(site)
        .with_state(engine)
        .layer(Extension(images))
}
