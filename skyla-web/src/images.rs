use gloo_net::http::Request;
use leptos::prelude::*;
use leptos::task::spawn_local;
use skyla_types::{ImageEntry, ImageModelEntry, ImageRequest, ImageStatus};
use std::time::Duration;

#[derive(Clone, Copy)]
pub struct Studio {
    pub models: RwSignal<Vec<ImageModelEntry>>,
    pub model: RwSignal<String>,
    pub gallery: RwSignal<Vec<ImageEntry>>,
    pub status: RwSignal<ImageStatus>,
    pub prompt: RwSignal<String>,
    pub negative: RwSignal<String>,
    pub steps: RwSignal<u32>,
    pub guidance: RwSignal<f32>,
    pub problem: RwSignal<Option<String>>,
}

async fn get<T: serde::de::DeserializeOwned>(url: &str) -> Option<T> {
    Request::get(url).send().await.ok()?.json().await.ok()
}

async fn message(resp: gloo_net::http::Response) -> String {
    let body: Option<serde_json::Value> = resp.json().await.ok();
    body.and_then(|v| v["message"].as_str().map(str::to_string))
        .unwrap_or_else(|| "request failed".into())
}

impl Studio {
    pub fn new() -> Self {
        Self {
            models: RwSignal::new(Vec::new()),
            model: RwSignal::new(String::new()),
            gallery: RwSignal::new(Vec::new()),
            status: RwSignal::new(ImageStatus::Idle),
            prompt: RwSignal::new(String::new()),
            negative: RwSignal::new("blurry, lowres, deformed, bad anatomy, watermark, text".into()),
            steps: RwSignal::new(20),
            guidance: RwSignal::new(7.0),
            problem: RwSignal::new(None),
        }
    }

    pub fn busy(&self) -> bool {
        self.status.with(ImageStatus::busy)
    }

    pub fn refresh(self) {
        spawn_local(async move {
            if let Some(models) = get::<Vec<ImageModelEntry>>("/api/images/models").await {
                if self.model.with_untracked(String::is_empty) {
                    if let Some(first) = models.iter().find(|m| m.xl).or(models.first()) {
                        self.model.set(first.id.clone());
                    }
                }
                self.models.set(models);
            }
            if let Some(gallery) = get::<Vec<ImageEntry>>("/api/images").await {
                self.gallery.set(gallery);
            }
            if let Some(status) = get::<ImageStatus>("/api/images/status").await {
                let busy = status.busy();
                self.status.set(status);
                if busy {
                    self.poll();
                }
            }
        });
    }

    pub fn generate(self) {
        let request = ImageRequest {
            model: self.model.get_untracked(),
            prompt: self.prompt.get_untracked(),
            negative: self.negative.get_untracked(),
            steps: self.steps.get_untracked(),
            guidance: self.guidance.get_untracked(),
        };
        self.problem.set(None);
        self.status.set(ImageStatus::Starting);
        spawn_local(async move {
            let sent = Request::post("/api/images/generate").json(&request).map(|r| r.send());
            match sent {
                Ok(fut) => match fut.await {
                    Ok(resp) if resp.ok() => self.poll(),
                    Ok(resp) => {
                        self.problem.set(Some(message(resp).await));
                        self.status.set(ImageStatus::Idle);
                    }
                    Err(e) => {
                        self.problem.set(Some(e.to_string()));
                        self.status.set(ImageStatus::Idle);
                    }
                },
                Err(e) => self.problem.set(Some(e.to_string())),
            }
        });
    }

    pub fn cancel(self) {
        spawn_local(async move {
            let _ = Request::post("/api/images/cancel").send().await;
        });
    }

    fn poll(self) {
        spawn_local(async move {
            loop {
                gloo_timers::future::sleep(Duration::from_millis(350)).await;
                let Some(status) = get::<ImageStatus>("/api/images/status").await else {
                    continue;
                };
                let busy = status.busy();
                if let ImageStatus::Error { message } = &status {
                    self.problem.set(Some(message.clone()));
                }
                self.status.set(status);
                if !busy {
                    break;
                }
            }
            if let Some(gallery) = get::<Vec<ImageEntry>>("/api/images").await {
                self.gallery.set(gallery);
            }
        });
    }
}
