use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Settings {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub max_tokens: usize,
    pub system: Option<String>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            temperature: 0.6,
            top_k: 40,
            top_p: 0.95,
            max_tokens: 4096,
            system: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientMessage {
    Chat { text: String, settings: Settings },
    Stop,
    Reset,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Info {
    pub model: String,
    pub device: String,
    pub context: usize,
    pub kv_cache: String,
    pub position: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Stats {
    pub prompt_tokens: usize,
    pub prefill_ms: f64,
    pub tokens: usize,
    pub tokens_per_second: f64,
    pub position: usize,
    pub context: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerMessage {
    Info(Info),
    Started,
    Token { text: String, thinking: bool },
    Done(Stats),
    Reset { position: usize },
    Error { message: String },
}
