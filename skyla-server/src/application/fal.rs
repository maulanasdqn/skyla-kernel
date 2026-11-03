use serde_json::{json, Value};
use skyla_types::{ImageEntry, ImageModelEntry, ImageRequest, ImageStatus};
use std::io::Read;
use std::path::Path;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

const MODELS: [(&str, &str); 4] = [
    ("fal-ai/flux/schnell", "FLUX.1 schnell · fal.ai"),
    ("fal-ai/flux/dev", "FLUX.1 dev · fal.ai"),
    ("fal-ai/flux-2-dev", "FLUX.2 dev · fal.ai"),
    ("fal-ai/flux-pro/v1.1-ultra", "FLUX 1.1 pro ultra · fal.ai"),
];

pub fn key() -> Option<String> {
    std::env::var("FAL_KEY").ok().filter(|k| !k.is_empty()).or_else(|| {
        let home = std::env::var("HOME").ok()?;
        let key = std::fs::read_to_string(Path::new(&home).join(".config/skyla/fal_key")).ok()?;
        Some(key.trim().to_string()).filter(|k| !k.is_empty())
    })
}

pub fn models() -> Vec<ImageModelEntry> {
    MODELS
        .iter()
        .map(|(id, name)| ImageModelEntry {
            id: id.to_string(),
            name: name.to_string(),
            xl: true,
            cloud: true,
        })
        .collect()
}

fn body(req: &ImageRequest, seed: u32) -> Value {
    if req.model.contains("flux-pro") {
        return json!({
            "prompt": req.prompt,
            "aspect_ratio": "1:1",
            "safety_tolerance": "5",
            "enable_safety_checker": false,
            "output_format": "png",
            "seed": seed,
        });
    }
    let steps = match req.model.as_str() {
        "fal-ai/flux/schnell" => req.steps.clamp(1, 12),
        _ => req.steps.clamp(4, 50),
    };
    json!({
        "prompt": req.prompt,
        "image_size": "square_hd",
        "num_inference_steps": steps,
        "guidance_scale": req.guidance.clamp(1.0, 10.0),
        "enable_safety_checker": false,
        "output_format": "png",
        "seed": seed,
    })
}

pub fn generate(key: &str, req: &ImageRequest, seed: u32, output: &Path) -> Result<bool, String> {
    let response: Value = ureq::post(&format!("https://fal.run/{}", req.model))
        .set("Authorization", &format!("Key {key}"))
        .send_json(body(req, seed))
        .map_err(|e| match e {
            ureq::Error::Status(code, resp) => {
                let text = resp.into_string().unwrap_or_default();
                format!("fal.ai returned {code}: {}", text.chars().take(300).collect::<String>())
            }
            other => format!("fal.ai request failed: {other}"),
        })?
        .into_json()
        .map_err(|e| format!("fal.ai response: {e}"))?;
    let url = response["images"][0]["url"]
        .as_str()
        .ok_or("fal.ai returned no image")?;
    let mut bytes = Vec::new();
    ureq::get(url)
        .call()
        .map_err(|e| format!("downloading the image: {e}"))?
        .into_reader()
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    std::fs::write(output, bytes).map_err(|e| e.to_string())?;
    let flagged = response["has_nsfw_concepts"]
        .as_array()
        .is_some_and(|a| a.iter().any(|v| v.as_bool() == Some(true)));
    Ok(flagged)
}

pub fn spawn(req: ImageRequest, seed: u32, output: PathBuf, status: Arc<Mutex<ImageStatus>>) {
    *status.lock().unwrap() = ImageStatus::Remote;
    std::thread::spawn(move || {
        let started = Instant::now();
        let next = match key()
            .ok_or_else(|| "no fal.ai key".to_string())
            .and_then(|k| generate(&k, &req, seed, &output))
        {
            Ok(true) => ImageStatus::Error {
                message: "fal.ai flagged this image as unsafe and blacked it out. Turning its safety checker off \
                          requires fal.ai to authorize your account."
                    .into(),
            },
            Ok(false) => ImageStatus::Done {
                image: ImageEntry {
                    name: output.file_name().unwrap_or_default().to_string_lossy().into_owned(),
                    url: format!("/images/{}", output.file_name().unwrap_or_default().to_string_lossy()),
                },
                seconds: started.elapsed().as_secs_f64(),
                sampling_seconds: started.elapsed().as_secs_f64(),
            },
            Err(message) => ImageStatus::Error { message },
        };
        *status.lock().unwrap() = next;
    });
}
