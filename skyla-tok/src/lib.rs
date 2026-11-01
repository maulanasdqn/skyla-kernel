mod bpe;

use anyhow::{anyhow, bail, Result};
use fancy_regex::Regex;
use skyla_gguf::{Gguf, Value};
use std::collections::HashMap;

const QWEN2_PRE: &str =
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const CONTROL: u64 = 3;
const USER_DEFINED: u64 = 4;

pub struct Tokenizer {
    vocab: Vec<String>,
    special: Vec<bool>,
    ids: HashMap<String, u32>,
    ranks: HashMap<(String, String), u32>,
    specials: Vec<(String, u32)>,
    byte_enc: [char; 256],
    byte_dec: HashMap<char, u8>,
    pre: Regex,
    pub bos: u32,
    pub eos: u32,
}

fn strings(g: &Gguf, key: &str) -> Result<Vec<String>> {
    let items = g
        .meta(key)?
        .as_array()
        .ok_or_else(|| anyhow!("{key} is not an array"))?;
    items
        .iter()
        .map(|v| {
            v.as_str()
                .map(str::to_owned)
                .ok_or_else(|| anyhow!("{key} holds a non-string"))
        })
        .collect()
}

impl Tokenizer {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let model = g.meta_str("tokenizer.ggml.model")?;
        let pre = g.meta_str("tokenizer.ggml.pre")?;
        if model != "gpt2" || pre != "qwen2" {
            bail!("unsupported tokenizer {model}/{pre} (only gpt2/qwen2)");
        }
        let vocab = strings(g, "tokenizer.ggml.tokens")?;
        let types = g
            .meta("tokenizer.ggml.token_type")?
            .as_array()
            .ok_or_else(|| anyhow!("token_type is not an array"))?;
        let special: Vec<bool> = types
            .iter()
            .map(|t| matches!(t.as_u64(), Some(CONTROL | USER_DEFINED)))
            .collect();
        let ids = vocab.iter().enumerate().map(|(i, s)| (s.clone(), i as u32)).collect();
        let mut ranks = HashMap::new();
        for (rank, line) in strings(g, "tokenizer.ggml.merges")?.into_iter().enumerate() {
            let (a, b) = line.split_once(' ').ok_or_else(|| anyhow!("bad merge {line:?}"))?;
            ranks.insert((a.to_owned(), b.to_owned()), rank as u32);
        }
        let mut specials: Vec<(String, u32)> = vocab
            .iter()
            .enumerate()
            .filter(|(i, _)| special[*i])
            .map(|(i, s)| (s.clone(), i as u32))
            .collect();
        specials.sort_by_key(|(s, _)| std::cmp::Reverse(s.len()));
        let byte_enc = bpe::byte_chars();
        let byte_dec = byte_enc.iter().enumerate().map(|(b, &c)| (c, b as u8)).collect();
        let id = |key: &str| g.meta(key).ok().and_then(Value::as_u64).map(|v| v as u32);
        Ok(Self {
            bos: id("tokenizer.ggml.bos_token_id").unwrap_or(0),
            eos: id("tokenizer.ggml.eos_token_id").unwrap_or(0),
            vocab,
            special,
            ids,
            ranks,
            specials,
            byte_enc,
            byte_dec,
            pre: Regex::new(QWEN2_PRE)?,
        })
    }

    pub fn id(&self, token: &str) -> Option<u32> {
        self.ids.get(token).copied()
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            let next = self
                .specials
                .iter()
                .filter_map(|(s, id)| rest.find(s.as_str()).map(|at| (at, s.len(), *id)))
                .min_by_key(|&(at, len, _)| (at, std::cmp::Reverse(len)));
            let (plain, special) = match next {
                Some((at, len, id)) => (&rest[..at], Some((len, id))),
                None => (rest, None),
            };
            self.encode_plain(plain, &mut out)?;
            match special {
                Some((len, id)) => {
                    out.push(id);
                    rest = &rest[plain.len() + len..];
                }
                None => break,
            }
        }
        Ok(out)
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) -> Result<()> {
        for piece in self.pre.find_iter(text) {
            let mapped: String = piece?.as_str().bytes().map(|b| self.byte_enc[b as usize]).collect();
            for symbol in bpe::merge(&mapped, &self.ranks) {
                match self.ids.get(&symbol) {
                    Some(&id) => out.push(id),
                    None => {
                        for c in symbol.chars() {
                            let id = self.ids.get(&c.to_string());
                            out.push(*id.ok_or_else(|| anyhow!("no byte token for {c:?}"))?);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub fn decode(&self, id: u32) -> Vec<u8> {
        let Some(token) = self.vocab.get(id as usize) else {
            return Vec::new();
        };
        match self.special[id as usize] {
            true => token.as_bytes().to_vec(),
            false => token
                .chars()
                .map(|c| self.byte_dec.get(&c).copied().unwrap_or(b'?'))
                .collect(),
        }
    }
}
