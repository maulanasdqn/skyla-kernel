use crate::domain::Job;
use anyhow::Result;
use skyla_chat::{seed_from_clock, Piece, Sampler, Session};
use skyla_engine::KvPrecision;
use skyla_gguf::Gguf;
use skyla_tok::Tokenizer;
use skyla_types::ServerMessage;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender};

fn open(model: PathBuf, context: usize, kv: KvPrecision) -> Result<Session<'static>> {
    let gguf: &'static Gguf = Box::leak(Box::new(Gguf::open(model)?));
    let tok: &'static Tokenizer = Box::leak(Box::new(Tokenizer::from_gguf(gguf)?));
    Session::new(gguf, tok, context, kv)
}

pub fn run(model: PathBuf, context: usize, kv: KvPrecision, inbox: Receiver<Job>, ready: Sender<Result<(), String>>) {
    let mut session = match open(model, context, kv) {
        Ok(s) => {
            let _ = ready.send(Ok(()));
            s
        }
        Err(e) => {
            let _ = ready.send(Err(format!("{e:#}")));
            return;
        }
    };

    while let Ok(job) = inbox.recv() {
        match job {
            Job::Info(reply) => {
                let _ = reply.send(session.info());
            }
            Job::Reset(reply) => {
                session.reset();
                let _ = reply.send(session.info().position);
            }
            Job::Chat {
                text,
                settings,
                events,
                cancel,
            } => {
                if events.send(ServerMessage::Started).is_err() {
                    continue;
                }
                let mut sampler = Sampler::from_settings(&settings, seed_from_clock());
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
                let message = match session.turn(&text, system, &mut sampler, settings.max_tokens, emit) {
                    Ok(stats) => ServerMessage::Done(stats),
                    Err(e) => ServerMessage::Error {
                        message: format!("{e:#}"),
                    },
                };
                let _ = events.send(message);
            }
        }
    }
}
