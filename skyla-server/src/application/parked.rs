use crate::domain::Job;
use skyla_types::{Info, ServerMessage};
use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;
use tokio::sync::oneshot;

pub type LoadReply = oneshot::Sender<Result<Info, String>>;

pub enum Next {
    Load {
        id: String,
        path: PathBuf,
        reply: Option<LoadReply>,
        pending: Option<Job>,
    },
    Unload(oneshot::Sender<Info>, Info),
    Quit,
}

pub fn idle(parked: &Info, path: &Path, inbox: &Receiver<Job>) -> Next {
    for job in inbox.iter() {
        match job {
            Job::Info(reply) | Job::Unload(reply) => {
                let _ = reply.send(parked.clone());
            }
            Job::Reset(reply) => {
                let _ = reply.send(0);
            }
            Job::Load { id, path, reply } => {
                return Next::Load {
                    id,
                    path,
                    reply: Some(reply),
                    pending: None,
                }
            }
            chat @ Job::Chat { .. } => {
                if let Job::Chat { events, .. } = &chat {
                    let _ = events.send(ServerMessage::Loading {
                        id: parked.model_id.clone(),
                    });
                }
                return Next::Load {
                    id: parked.model_id.clone(),
                    path: path.to_path_buf(),
                    reply: None,
                    pending: Some(chat),
                };
            }
        }
    }
    Next::Quit
}

pub fn fail(pending: Option<Job>, message: &str) {
    if let Some(Job::Chat { events, .. }) = pending {
        let _ = events.send(ServerMessage::Error {
            message: message.to_string(),
        });
    }
}
