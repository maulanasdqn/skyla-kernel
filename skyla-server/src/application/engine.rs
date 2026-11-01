use crate::application::catalog::Catalog;
use crate::application::worker;
use crate::domain::{Cancel, Job};
use crate::error::AppError;
use anyhow::{anyhow, Result};
use skyla_engine::KvPrecision;
use skyla_types::{Info, ModelEntry, ServerMessage, Settings};
use std::path::PathBuf;
use std::sync::{mpsc, Arc};
use tokio::sync::{mpsc as async_mpsc, oneshot};

#[derive(Clone)]
pub struct EngineHandle {
    jobs: mpsc::Sender<Job>,
    catalog: Arc<Catalog>,
}

fn unavailable<T>(_: T) -> AppError {
    AppError::Unavailable("the engine thread has stopped".into())
}

impl EngineHandle {
    pub fn spawn(model: PathBuf, catalog: Catalog, context: usize, kv: KvPrecision) -> Result<Self> {
        let (jobs, inbox) = mpsc::channel();
        let (ready_tx, ready_rx) = mpsc::channel();
        let start = (catalog.id_of(&model), model);
        std::thread::Builder::new()
            .name("skyla-engine".into())
            .spawn(move || worker::run(start, context, kv, inbox, ready_tx))?;
        ready_rx
            .recv()
            .map_err(|_| anyhow!("engine thread exited during startup"))?
            .map_err(|e| anyhow!(e))?;
        Ok(Self {
            jobs,
            catalog: Arc::new(catalog),
        })
    }

    fn send(&self, job: Job) -> Result<(), AppError> {
        self.jobs.send(job).map_err(unavailable)
    }

    pub fn models(&self) -> Vec<ModelEntry> {
        self.catalog.models()
    }

    pub async fn info(&self) -> Result<Info, AppError> {
        let (tx, rx) = oneshot::channel();
        self.send(Job::Info(tx))?;
        rx.await.map_err(unavailable)
    }

    pub async fn reset(&self) -> Result<usize, AppError> {
        let (tx, rx) = oneshot::channel();
        self.send(Job::Reset(tx))?;
        rx.await.map_err(unavailable)
    }

    pub async fn load(&self, id: &str) -> Result<Info, AppError> {
        let (entry, path) = self
            .catalog
            .find(id)
            .ok_or_else(|| AppError::BadRequest(format!("unknown model {id}")))?;
        if !entry.supported {
            return Err(AppError::BadRequest(format!(
                "{} uses the {} architecture, which Skyla does not support yet",
                entry.name, entry.arch
            )));
        }
        let (tx, rx) = oneshot::channel();
        self.send(Job::Load {
            id: entry.id,
            path,
            reply: tx,
        })?;
        rx.await.map_err(unavailable)?.map_err(AppError::Unavailable)
    }

    pub fn chat(
        &self,
        text: String,
        settings: Settings,
        events: async_mpsc::UnboundedSender<ServerMessage>,
        cancel: Cancel,
    ) -> Result<(), AppError> {
        self.send(Job::Chat {
            text,
            settings,
            events,
            cancel,
        })
    }
}
