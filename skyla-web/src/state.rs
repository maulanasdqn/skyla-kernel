use leptos::prelude::*;
use skyla_types::{Info, ModelEntry, ServerMessage, Settings, Stats};

#[derive(Clone, Copy)]
pub struct Turn {
    pub id: usize,
    pub user: bool,
    pub text: RwSignal<String>,
    pub thinking: RwSignal<String>,
    pub stats: RwSignal<Option<Stats>>,
    pub error: RwSignal<Option<String>>,
}

impl Turn {
    fn new(id: usize, user: bool, text: String) -> Self {
        Self {
            id,
            user,
            text: RwSignal::new(text),
            thinking: RwSignal::new(String::new()),
            stats: RwSignal::new(None),
            error: RwSignal::new(None),
        }
    }
}

#[derive(Clone, Copy)]
pub struct Chat {
    pub turns: RwSignal<Vec<Turn>>,
    pub info: RwSignal<Option<Info>>,
    pub models: RwSignal<Vec<ModelEntry>>,
    pub loading: RwSignal<Option<String>>,
    pub busy: RwSignal<bool>,
    pub connected: RwSignal<bool>,
    pub settings: RwSignal<Settings>,
    pub tick: RwSignal<u64>,
    next_id: StoredValue<usize>,
}

impl Chat {
    pub fn new() -> Self {
        Self {
            turns: RwSignal::new(Vec::new()),
            info: RwSignal::new(None),
            models: RwSignal::new(Vec::new()),
            loading: RwSignal::new(None),
            busy: RwSignal::new(false),
            connected: RwSignal::new(false),
            settings: RwSignal::new(Settings::default()),
            tick: RwSignal::new(0),
            next_id: StoredValue::new(0),
        }
    }

    pub fn push(&self, user: bool, text: String) {
        let id = self.next_id.get_value();
        self.next_id.set_value(id + 1);
        self.turns.update(|t| t.push(Turn::new(id, user, text)));
    }

    fn last_reply(&self) -> Option<Turn> {
        self.turns.with_untracked(|t| t.last().copied()).filter(|t| !t.user)
    }

    fn set_position(&self, position: usize) {
        self.info.update(|i| {
            if let Some(i) = i {
                i.position = position;
            }
        });
    }

    pub fn apply(&self, message: ServerMessage) {
        match message {
            ServerMessage::Models { models } => self.models.set(models),
            ServerMessage::Loading { id } => {
                self.loading.set(Some(id));
                self.busy.set(false);
            }
            ServerMessage::Info(info) => {
                let changed = self
                    .info
                    .with_untracked(|i| i.as_ref().is_some_and(|i| i.model_id != info.model_id));
                if changed {
                    self.turns.set(Vec::new());
                }
                self.loading.set(None);
                self.info.set(Some(info));
            }
            ServerMessage::Started => self.busy.set(true),
            ServerMessage::Token { text, thinking } => {
                if let Some(turn) = self.last_reply() {
                    let target = if thinking { turn.thinking } else { turn.text };
                    target.update(|s| s.push_str(&text));
                    self.tick.update(|n| *n += 1);
                }
            }
            ServerMessage::Done(stats) => {
                self.set_position(stats.position);
                if let Some(turn) = self.last_reply() {
                    turn.stats.set(Some(stats));
                }
                self.busy.set(false);
            }
            ServerMessage::Reset { position } => {
                self.turns.set(Vec::new());
                self.set_position(position);
                self.busy.set(false);
            }
            ServerMessage::Error { message } => {
                self.loading.set(None);
                match self.last_reply() {
                    Some(turn) => turn.error.set(Some(message)),
                    None => {
                        self.push(false, String::new());
                        if let Some(turn) = self.last_reply() {
                            turn.error.set(Some(message));
                        }
                    }
                }
                self.busy.set(false);
            }
        }
    }
}
