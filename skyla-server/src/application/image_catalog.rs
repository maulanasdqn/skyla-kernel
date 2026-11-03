use skyla_types::{ImageEntry, ImageModelEntry};
use std::path::Path;

pub(super) fn model_name(id: &str) -> String {
    match id {
        "sd15" => "Stable Diffusion 1.5".into(),
        "sdxl" => "Stable Diffusion XL".into(),
        other => other.replace(['-', '_'], " "),
    }
}

pub(super) fn entry(path: &Path) -> Option<ImageEntry> {
    let name = path.file_name()?.to_str()?.to_string();
    Some(ImageEntry {
        url: format!("/images/{name}"),
        name,
    })
}

pub(super) fn local_models(dir: &Path) -> Vec<ImageModelEntry> {
    let mut out: Vec<ImageModelEntry> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.join("TextEncoder.mlmodelc").exists())
        .filter_map(|p| {
            let id = p.file_name()?.to_str()?.to_string();
            Some(ImageModelEntry {
                name: model_name(&id),
                xl: p.join("TextEncoder2.mlmodelc").exists(),
                cloud: false,
                id,
            })
        })
        .collect();
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out
}
