use crate::application::worker;
use crate::domain::{Cancel, Job};
use crate::error::AppError;
use anyhow::{anyhow, Result};
use skyla_engine::KvPrecision;
use skyla_types::{Info, ServerMessage, Settings};
use std::path::PathBuf;
use std::sync::mpsc;
use tokio::sync::{mpsc as async_mpsc, oneshot};

#[derive(Clone)]
pub struct EngineHandle {
    jobs: mpsc::Sender<Job>,
}

impl EngineHandle {
    pub fn spawn(model: PathBuf, context: usize, kv: KvPrecision) -> Result<Self> {
        let (jobs, inbox) = mpsc::channel();
        let (ready_tx, ready_rx) = mpsc::channel();
        std::thread::Builder::new()
            .name("skyla-engine".into())
            .spawn(move || worker::run(model, context, kv, inbox, ready_tx))?;
        ready_rx
            .recv()
            .map_err(|_| anyhow!("engine thread exited during startup"))?
            .map_err(|e| anyhow!(e))?;
        Ok(Self { jobs })
    }

    fn send(&self, job: Job) -> Result<(), AppError> {
        self.jobs
            .send(job)
            .map_err(|_| AppError::Unavailable("the engine thread has stopped".into()))
    }

    pub async fn info(&self) -> Result<Info, AppError> {
        let (tx, rx) = oneshot::channel();
        self.send(Job::Info(tx))?;
        rx.await
            .map_err(|_| AppError::Unavailable("the engine did not answer".into()))
    }

    pub async fn reset(&self) -> Result<usize, AppError> {
        let (tx, rx) = oneshot::channel();
        self.send(Job::Reset(tx))?;
        rx.await
            .map_err(|_| AppError::Unavailable("the engine did not answer".into()))
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
