use crate::application::EngineHandle;
use crate::error::AppError;
use axum::extract::State;
use axum::Json;
use skyla_types::Info;

pub async fn info(State(engine): State<EngineHandle>) -> Result<Json<Info>, AppError> {
    Ok(Json(engine.info().await?))
}

pub async fn reset(State(engine): State<EngineHandle>) -> Result<Json<Info>, AppError> {
    engine.reset().await?;
    Ok(Json(engine.info().await?))
}
