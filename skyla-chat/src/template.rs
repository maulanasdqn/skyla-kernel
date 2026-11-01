use skyla_gguf::Gguf;
use skyla_tok::Tokenizer;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Template {
    DeepSeek,
    ChatMl,
}

impl Template {
    pub fn detect(g: &Gguf) -> Self {
        match g.meta_str("tokenizer.chat_template") {
            Ok(t) if t.contains("<|im_start|>") => Self::ChatMl,
            _ => Self::DeepSeek,
        }
    }

    pub fn stop_ids(self, tok: &Tokenizer) -> Vec<u32> {
        let mut ids = vec![tok.eos];
        for name in ["<|im_end|>", "<|endoftext|>", "<｜end▁of▁sentence｜>"] {
            ids.extend(tok.id(name));
        }
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    fn end_of_reply(self) -> &'static str {
        match self {
            Self::DeepSeek => "<｜end▁of▁sentence｜>",
            Self::ChatMl => "<|im_end|>",
        }
    }

    pub fn turn(self, bos: &str, first: bool, open_reply: bool, system: Option<&str>, user: &str) -> String {
        let mut prompt = String::new();
        if open_reply {
            prompt.push_str(self.end_of_reply());
        }
        match self {
            Self::DeepSeek => {
                if first {
                    prompt.push_str(bos);
                    prompt.push_str(system.unwrap_or(""));
                }
                prompt.push_str(&format!("<｜User｜>{user}<｜Assistant｜>"));
            }
            Self::ChatMl => {
                if !first {
                    prompt.push('\n');
                }
                if let Some(system) = system.filter(|_| first) {
                    prompt.push_str(&format!("<|im_start|>system\n{system}<|im_end|>\n"));
                }
                prompt.push_str(&format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"));
            }
        }
        prompt
    }
}
