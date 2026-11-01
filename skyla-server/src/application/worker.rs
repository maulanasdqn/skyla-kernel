use crate::domain::{Cancel, Job};
use anyhow::Result;
use skyla_chat::{seed_from_clock, Piece, Sampler, Session};
use skyla_engine::KvPrecision;
use skyla_gguf::Gguf;
use skyla_tok::Tokenizer;
use skyla_types::{Info, ServerMessage, Settings};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender};
use tokio::sync::{mpsc, oneshot};

type LoadReply = oneshot::Sender<Result<Info, String>>;

enum Next {
    Load {
        id: String,
        path: PathBuf,
        reply: LoadReply,
    },
    Quit,
}

fn info(session: &Session, id: &str) -> Info {
    Info {
        model_id: id.to_string(),
        ..session.info()
    }
}

fn chat(
    session: &mut Session,
    text: &str,
    settings: &Settings,
    events: &mpsc::UnboundedSender<ServerMessage>,
    cancel: &Cancel,
) {
    if events.send(ServerMessage::Started).is_err() {
        return;
    }
    let mut sampler = Sampler::from_settings(settings, seed_from_clock());
    let system = settings.system.as_deref().filter(|s| !s.trim().is_empty());
    let emit = |piece: Piece| {
        if cancel.is_set() {
            return false;
        }
        match piece {
            Piece::Text { text, thinking } => events.send(ServerMessage::Token { text, thinking }).is_ok(),
            Piece::ThinkStart | Piece::ThinkEnd => true,
        }
    };
    let message = match session.turn(text, system, &mut sampler, settings.max_tokens, emit) {
        Ok(stats) => ServerMessage::Done(stats),
        Err(e) => ServerMessage::Error {
            message: format!("{e:#}"),
        },
    };
    let _ = events.send(message);
}

fn serve(session: &mut Session, id: &str, inbox: &Receiver<Job>) -> Next {
    while let Ok(job) = inbox.recv() {
        match job {
            Job::Info(reply) => {
                let _ = reply.send(info(session, id));
            }
            Job::Reset(reply) => {
                session.reset();
                let _ = reply.send(0);
            }
            Job::Chat {
                text,
                settings,
                events,
                cancel,
            } => chat(session, &text, &settings, &events, &cancel),
            Job::Load { id, path, reply } => return Next::Load { id, path, reply },
        }
    }
    Next::Quit
}

fn with_model(path: &Path, context: usize, kv: KvPrecision, body: impl FnOnce(&mut Session) -> Next) -> Result<Next> {
    let gguf = Gguf::open(path)?;
    let tok = Tokenizer::from_gguf(&gguf)?;
    let mut session = Session::new(&gguf, &tok, context, kv)?;
    Ok(body(&mut session))
}

pub fn run(
    start: (String, PathBuf),
    context: usize,
    kv: KvPrecision,
    inbox: Receiver<Job>,
    ready: Sender<Result<(), String>>,
) {
    let mut current = start.clone();
    let mut previous = start;
    let mut ready = Some(ready);
    let mut reply: Option<LoadReply> = None;
    let mut falling_back = false;
    loop {
        let (id, path) = current.clone();
        let result = with_model(&path, context, kv, |session| {
            if let Some(r) = ready.take() {
                let _ = r.send(Ok(()));
            }
            if let Some(r) = reply.take() {
                let _ = r.send(Ok(info(session, &id)));
            }
            serve(session, &id, &inbox)
        });
        match result {
            Ok(Next::Load { id, path, reply: r }) => {
                previous = current;
                current = (id, path);
                reply = Some(r);
                falling_back = false;
            }
            Ok(Next::Quit) => return,
            Err(e) => {
                let message = format!("{e:#}");
                if let Some(r) = ready.take() {
                    let _ = r.send(Err(message));
                    return;
                }
                if let Some(r) = reply.take() {
                    let _ = r.send(Err(message));
                }
                if falling_back {
                    return;
                }
                falling_back = true;
                current = previous.clone();
            }
        }
    }
}
