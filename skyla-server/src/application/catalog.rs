use skyla_gguf::Gguf;
use skyla_types::ModelEntry;
use std::path::{Path, PathBuf};

const SUPPORTED: [&str; 2] = ["qwen3", "qwen3moe"];

#[derive(Clone, Default)]
pub struct Catalog {
    entries: Vec<(ModelEntry, PathBuf)>,
}

fn files(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
    let Ok(read) = std::fs::read_dir(dir) else { return };
    for entry in read.flatten() {
        let path = entry.path();
        if path.is_dir() && depth > 0 {
            files(&path, depth - 1, out);
        } else if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("gguf")) {
            out.push(path);
        }
    }
}

fn describe(root: &Path, path: &Path) -> ModelEntry {
    let id = path
        .strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/");
    let size_gb = std::fs::metadata(path).map_or(0.0, |m| m.len() as f64 / 1e9);
    let (name, arch) = match Gguf::open(path) {
        Ok(g) => (
            g.meta_str("general.name").unwrap_or(&id).to_string(),
            g.meta_str("general.architecture").unwrap_or("unknown").to_string(),
        ),
        Err(_) => (id.clone(), "unreadable".to_string()),
    };
    ModelEntry {
        supported: SUPPORTED.contains(&arch.as_str()),
        id,
        name,
        arch,
        size_gb,
    }
}

impl Catalog {
    pub fn scan(root: &Path) -> Self {
        let mut paths = Vec::new();
        files(root, 2, &mut paths);
        paths.sort();
        Self {
            entries: paths.into_iter().map(|p| (describe(root, &p), p)).collect(),
        }
    }

    pub fn models(&self) -> Vec<ModelEntry> {
        self.entries.iter().map(|(e, _)| e.clone()).collect()
    }

    pub fn find(&self, id: &str) -> Option<(ModelEntry, PathBuf)> {
        self.entries.iter().find(|(e, _)| e.id == id).cloned()
    }

    pub fn id_of(&self, path: &Path) -> String {
        let wanted = path.canonicalize().ok();
        self.entries
            .iter()
            .find(|(_, p)| p.canonicalize().ok() == wanted)
            .map_or_else(|| path.to_string_lossy().into_owned(), |(e, _)| e.id.clone())
    }
}
