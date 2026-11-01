use skyla_types::{Info, ServerMessage, Settings};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};

#[derive(Clone, Default)]
pub struct Cancel(Arc<AtomicBool>);

impl Cancel {
    pub fn set(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    pub fn clear(&self) {
        self.0.store(false, Ordering::Relaxed);
    }

    pub fn is_set(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

pub enum Job {
    Info(oneshot::Sender<Info>),
    Reset(oneshot::Sender<usize>),
    Chat {
        text: String,
        settings: Settings,
        events: mpsc::UnboundedSender<ServerMessage>,
        cancel: Cancel,
    },
}
