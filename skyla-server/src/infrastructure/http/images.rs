use crate::application::ImageService;
use crate::error::AppError;
use axum::{Extension, Json};
use skyla_types::{ImageEntry, ImageModelEntry, ImageRequest, ImageStatus};

pub async fn models(Extension(images): Extension<ImageService>) -> Json<Vec<ImageModelEntry>> {
    Json(images.models())
}

pub async fn gallery(Extension(images): Extension<ImageService>) -> Json<Vec<ImageEntry>> {
    Json(images.gallery())
}

pub async fn status(Extension(images): Extension<ImageService>) -> Json<ImageStatus> {
    Json(images.status())
}

pub async fn generate(
    Extension(images): Extension<ImageService>,
    Json(request): Json<ImageRequest>,
) -> Result<Json<ImageStatus>, AppError> {
    images.generate(request)?;
    Ok(Json(images.status()))
}

pub async fn cancel(Extension(images): Extension<ImageService>) -> Json<ImageStatus> {
    images.cancel();
    Json(images.status())
}
