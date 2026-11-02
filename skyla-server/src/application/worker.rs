use crate::application::parked::{fail, idle, LoadReply, Next};
use crate::domain::{Cancel, Job};
use anyhow::Result;
use skyla_chat::{seed_from_clock, Piece, Sampler, Session};
use skyla_engine::KvPrecision;
use skyla_gguf::Gguf;
use skyla_tok::Tokenizer;
use skyla_types::{Info, ServerMessage, Settings};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender};
use tokio::sync::mpsc;

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

fn serve(session: &mut Session, id: &str, first: Option<Job>, inbox: &Receiver<Job>) -> Next {
    for job in first.into_iter().chain(inbox.iter()) {
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
            Job::Unload(reply) => {
                let parked = Info {
                    position: 0,
                    loaded: false,
                    ..info(session, id)
                };
                return Next::Unload(reply, parked);
            }
            Job::Load { id, path, reply } => {
                return Next::Load {
                    id,
                    path,
                    reply: Some(reply),
                    pending: None,
                }
            }
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
    let mut pending: Option<Job> = None;
    let mut parked: Option<Info> = None;
    let mut falling_back = false;
    loop {
        let (id, path) = current.clone();
        let result = match &parked {
            Some(info) => Ok(idle(info, &path, &inbox)),
            None => with_model(&path, context, kv, |session| {
                if let Some(r) = ready.take() {
                    let _ = r.send(Ok(()));
                }
                if let Some(r) = reply.take() {
                    let _ = r.send(Ok(info(session, &id)));
                }
                if let Some(Job::Chat { events, .. }) = &pending {
                    let _ = events.send(ServerMessage::Info(info(session, &id)));
                }
                serve(session, &id, pending.take(), &inbox)
            }),
        };
        match result {
            Ok(Next::Load {
                id,
                path,
                reply: r,
                pending: p,
            }) => {
                if parked.take().is_none() {
                    previous = current;
                }
                current = (id, path);
                reply = r;
                pending = p;
                falling_back = false;
            }
            Ok(Next::Unload(r, info)) => {
                let _ = r.send(info.clone());
                parked = Some(info);
            }
            Ok(Next::Quit) => return,
            Err(e) => {
                let message = format!("{e:#}");
                if let Some(r) = ready.take() {
                    let _ = r.send(Err(message));
                    return;
                }
                if let Some(r) = reply.take() {
                    let _ = r.send(Err(message.clone()));
                }
                fail(pending.take(), &message);
                if falling_back {
                    return;
                }
                falling_back = true;
                current = previous.clone();
            }
        }
    }
}
