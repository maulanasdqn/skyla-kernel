use super::fal;
use super::image_catalog::{entry, local_models};
use crate::error::AppError;
use serde_json::{json, Value};
use skyla_types::{ImageEntry, ImageModelEntry, ImageRequest, ImageStatus};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{ChildStdin, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone)]
pub struct ImageService {
    inner: Arc<Inner>,
}

struct Inner {
    models: PathBuf,
    outputs: PathBuf,
    tool: PathBuf,
    status: Arc<Mutex<ImageStatus>>,
    helper: Mutex<Option<ChildStdin>>,
}

impl ImageService {
    pub fn new(models: PathBuf, tool: PathBuf) -> Self {
        let outputs = models.join("outputs");
        Self {
            inner: Arc::new(Inner {
                models,
                outputs,
                tool,
                status: Arc::new(Mutex::new(ImageStatus::Idle)),
                helper: Mutex::new(None),
            }),
        }
    }

    pub fn outputs(&self) -> &Path {
        &self.inner.outputs
    }

    pub fn models(&self) -> Vec<ImageModelEntry> {
        let mut out = local_models(&self.inner.models);
        if fal::key().is_some() {
            out.extend(fal::models());
        }
        out
    }

    pub fn gallery(&self) -> Vec<ImageEntry> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(&self.inner.outputs)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "png"))
            .collect();
        files.sort();
        files.iter().rev().filter_map(|p| entry(p)).collect()
    }

    pub fn status(&self) -> ImageStatus {
        self.inner.status.lock().unwrap().clone()
    }

    pub fn generate(&self, req: ImageRequest) -> Result<(), AppError> {
        if req.prompt.trim().is_empty() {
            return Err(AppError::BadRequest("empty prompt".into()));
        }
        let model = self
            .models()
            .into_iter()
            .find(|m| m.id == req.model)
            .ok_or_else(|| AppError::BadRequest(format!("unknown image model {}", req.model)))?;
        {
            let mut status = self.inner.status.lock().unwrap();
            if status.busy() {
                return Err(AppError::BadRequest("an image is already being generated".into()));
            }
            *status = ImageStatus::Starting;
        }
        std::fs::create_dir_all(&self.inner.outputs).map_err(|e| AppError::Unavailable(e.to_string()))?;
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
        let seed = (now.as_nanos() as u64 ^ (now.as_nanos() >> 32) as u64) as u32;
        let output = self.inner.outputs.join(format!("{}-{seed}.png", now.as_secs()));
        if model.cloud {
            fal::spawn(req, seed, output, Arc::clone(&self.inner.status));
            return Ok(());
        }
        let job = json!({ "job": {
            "resources": self.inner.models.join(&model.id),
            "xl": model.xl,
            "prompt": req.prompt,
            "negative": req.negative,
            "steps": req.steps.clamp(4, 50),
            "guidance": req.guidance.clamp(1.0, 15.0),
            "seed": seed,
            "output": output,
        }});
        self.send(&job).map_err(|e| {
            *self.inner.status.lock().unwrap() = ImageStatus::Error { message: e.clone() };
            AppError::Unavailable(e)
        })
    }

    pub fn cancel(&self) {
        let _ = self.send(&json!({ "cancel": true }));
    }

    fn send(&self, message: &Value) -> Result<(), String> {
        let mut helper = self.inner.helper.lock().unwrap();
        if helper.is_none() {
            *helper = Some(self.spawn()?);
        }
        let stdin = helper.as_mut().unwrap();
        let line = format!("{message}\n");
        if stdin.write_all(line.as_bytes()).and_then(|_| stdin.flush()).is_err() {
            *helper = Some(self.spawn()?);
            let stdin = helper.as_mut().unwrap();
            stdin.write_all(line.as_bytes()).map_err(|e| e.to_string())?;
            stdin.flush().map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    fn spawn(&self) -> Result<ChildStdin, String> {
        let mut child = Command::new(&self.inner.tool)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .map_err(|e| format!("starting {}: {e} (build skyla-image first)", self.inner.tool.display()))?;
        let stdout = child.stdout.take().ok_or("helper has no stdout")?;
        let inner = Arc::clone(&self.inner);
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                let Ok(v) = serde_json::from_str::<Value>(&line) else {
                    continue;
                };
                let next = match v["type"].as_str() {
                    Some("loading") => ImageStatus::Loading,
                    Some("step") => ImageStatus::Step {
                        step: v["step"].as_u64().unwrap_or(0) as u32,
                        total: v["total"].as_u64().unwrap_or(0) as u32,
                    },
                    Some("done") => match v["output"].as_str().and_then(|p| entry(Path::new(p))) {
                        Some(image) => ImageStatus::Done {
                            image,
                            seconds: v["seconds"].as_f64().unwrap_or(0.0),
                            sampling_seconds: v["sampling_seconds"].as_f64().unwrap_or(0.0),
                        },
                        None => continue,
                    },
                    Some("cancelled") => ImageStatus::Cancelled,
                    Some("error") => ImageStatus::Error {
                        message: v["message"].as_str().unwrap_or("image generation failed").into(),
                    },
                    _ => continue,
                };
                *inner.status.lock().unwrap() = next;
            }
            let mut status = inner.status.lock().unwrap();
            if status.busy() {
                *status = ImageStatus::Error {
                    message: "the image helper stopped".into(),
                };
            }
            *inner.helper.lock().unwrap() = None;
        });
        child.stdin.take().ok_or_else(|| "helper has no stdin".into())
    }
}
