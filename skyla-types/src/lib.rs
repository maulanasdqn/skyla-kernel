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
    Load { id: String },
    Unload,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelEntry {
    pub id: String,
    pub name: String,
    pub arch: String,
    pub size_gb: f64,
    pub supported: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Info {
    pub model_id: String,
    pub model: String,
    pub device: String,
    pub context: usize,
    pub kv_cache: String,
    pub position: usize,
    pub loaded: bool,
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
    Models { models: Vec<ModelEntry> },
    Loading { id: String },
    Info(Info),
    Started,
    Token { text: String, thinking: bool },
    Done(Stats),
    Reset { position: usize },
    Error { message: String },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImageModelEntry {
    pub id: String,
    pub name: String,
    pub xl: bool,
    #[serde(default)]
    pub cloud: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImageEntry {
    pub url: String,
    pub name: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImageRequest {
    pub model: String,
    pub prompt: String,
    pub negative: String,
    pub steps: u32,
    pub guidance: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ImageStatus {
    Idle,
    Starting,
    Loading,
    Remote,
    Step {
        step: u32,
        total: u32,
    },
    Done {
        image: ImageEntry,
        seconds: f64,
        sampling_seconds: f64,
    },
    Cancelled,
    Error {
        message: String,
    },
}

impl ImageStatus {
    pub fn busy(&self) -> bool {
        matches!(self, Self::Starting | Self::Loading | Self::Remote | Self::Step { .. })
    }
}
