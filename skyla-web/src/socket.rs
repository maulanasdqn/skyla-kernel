use crate::state::Chat;
use futures::channel::mpsc;
use futures::{SinkExt, StreamExt};
use gloo_net::websocket::futures::WebSocket;
use gloo_net::websocket::Message;
use leptos::prelude::*;
use leptos::task::spawn_local;
use skyla_types::{ClientMessage, ServerMessage};

#[derive(Clone, Copy)]
pub struct Link(StoredValue<mpsc::UnboundedSender<ClientMessage>>);

impl Link {
    pub fn send(&self, message: ClientMessage) {
        self.0.with_value(|tx| {
            let _ = tx.unbounded_send(message);
        });
    }
}

fn url() -> String {
    let location = window().location();
    let host = location.host().unwrap_or_default();
    let secure = location.protocol().map(|p| p == "https:").unwrap_or(false);
    format!("{}://{host}/api/ws", if secure { "wss" } else { "ws" })
}

pub fn connect(chat: Chat) -> Link {
    let (tx, mut outbox) = mpsc::unbounded::<ClientMessage>();
    let link = Link(StoredValue::new(tx));
    let Ok(socket) = WebSocket::open(&url()) else {
        return link;
    };
    chat.connected.set(true);
    let (mut write, mut read) = socket.split();

    spawn_local(async move {
        while let Some(message) = outbox.next().await {
            let Ok(json) = serde_json::to_string(&message) else {
                continue;
            };
            if write.send(Message::Text(json)).await.is_err() {
                break;
            }
        }
    });

    spawn_local(async move {
        while let Some(frame) = read.next().await {
            match frame {
                Ok(Message::Text(text)) => {
                    if let Ok(message) = serde_json::from_str::<ServerMessage>(&text) {
                        chat.apply(message);
                    }
                }
                Ok(Message::Bytes(_)) => {}
                Err(_) => break,
            }
        }
        chat.connected.set(false);
        chat.busy.set(false);
    });
    link
}
