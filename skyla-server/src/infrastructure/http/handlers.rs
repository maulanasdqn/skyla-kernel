use crate::application::EngineHandle;
use crate::error::AppError;
use axum::extract::State;
use axum::Json;
use skyla_types::{Info, ModelEntry};

pub async fn info(State(engine): State<EngineHandle>) -> Result<Json<Info>, AppError> {
    Ok(Json(engine.info().await?))
}

pub async fn models(State(engine): State<EngineHandle>) -> Json<Vec<ModelEntry>> {
    Json(engine.models())
}

pub async fn reset(State(engine): State<EngineHandle>) -> Result<Json<Info>, AppError> {
    engine.reset().await?;
    Ok(Json(engine.info().await?))
}
