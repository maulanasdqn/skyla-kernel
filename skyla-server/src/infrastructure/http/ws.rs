use crate::application::EngineHandle;
use crate::domain::Cancel;
use crate::error::AppError;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::Response;
use futures_util::{SinkExt, StreamExt};
use skyla_types::{ClientMessage, ServerMessage};
use tokio::sync::mpsc;

pub async fn upgrade(ws: WebSocketUpgrade, State(engine): State<EngineHandle>) -> Response {
    ws.on_upgrade(move |socket| session(socket, engine))
}

fn encode(message: &ServerMessage) -> Option<Message> {
    serde_json::to_string(message).ok().map(|s| Message::Text(s.into()))
}

async fn handle(
    message: ClientMessage,
    engine: &EngineHandle,
    events: &mpsc::UnboundedSender<ServerMessage>,
    cancel: &Cancel,
) -> Result<(), AppError> {
    match message {
        ClientMessage::Chat { text, settings } => {
            if text.trim().is_empty() {
                return Err(AppError::BadRequest("empty message".into()));
            }
            cancel.clear();
            engine.chat(text, settings, events.clone(), cancel.clone())
        }
        ClientMessage::Stop => {
            cancel.set();
            Ok(())
        }
        ClientMessage::Reset => {
            cancel.set();
            let position = engine.reset().await?;
            let _ = events.send(ServerMessage::Reset { position });
            Ok(())
        }
        ClientMessage::Unload => {
            cancel.set();
            let info = engine.unload().await?;
            let _ = events.send(ServerMessage::Info(info));
            Ok(())
        }
        ClientMessage::Load { id } => {
            cancel.set();
            let _ = events.send(ServerMessage::Loading { id: id.clone() });
            let info = engine.load(&id).await?;
            let _ = events.send(ServerMessage::Info(info));
            Ok(())
        }
    }
}

async fn session(socket: WebSocket, engine: EngineHandle) {
    let (mut sink, mut stream) = socket.split();
    let (events, mut outbox) = mpsc::unbounded_channel::<ServerMessage>();
    let cancel = Cancel::default();

    let _ = events.send(ServerMessage::Models {
        models: engine.models(),
    });
    if let Ok(info) = engine.info().await {
        let _ = events.send(ServerMessage::Info(info));
    }
    let writer = tokio::spawn(async move {
        while let Some(message) = outbox.recv().await {
            let Some(frame) = encode(&message) else { continue };
            if sink.send(frame).await.is_err() {
                break;
            }
        }
    });

    while let Some(Ok(frame)) = stream.next().await {
        let Message::Text(text) = frame else { continue };
        let result = match serde_json::from_str::<ClientMessage>(&text) {
            Ok(message) => handle(message, &engine, &events, &cancel).await,
            Err(e) => Err(AppError::BadRequest(e.to_string())),
        };
        if let Err(AppError::BadRequest(message) | AppError::Unavailable(message)) = result {
            let _ = events.send(ServerMessage::Error { message });
        }
    }

    cancel.set();
    writer.abort();
}
