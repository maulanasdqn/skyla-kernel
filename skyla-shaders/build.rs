use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, fs};

fn glslc() -> PathBuf {
    if let Ok(p) = env::var("GLSLC") {
        return p.into();
    }
    if let Ok(sdk) = env::var("VULKAN_SDK") {
        for p in [
            Path::new(&sdk).join("Bin").join("glslc.exe"),
            Path::new(&sdk).join("bin").join("glslc"),
        ] {
            if p.exists() {
                return p;
            }
        }
    }
    "glslc".into()
}

fn main() {
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let dir = Path::new("shaders");
    println!("cargo:rerun-if-changed=shaders");
    println!("cargo:rerun-if-env-changed=GLSLC");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");

    let mut shaders: Vec<_> = fs::read_dir(dir)
        .expect("reading shaders/")
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "comp"))
        .collect();
    shaders.sort();

    let glslc = glslc();
    let mut consts = String::new();
    for src in &shaders {
        let stem = src.file_stem().unwrap().to_str().unwrap();
        let variants: &[(&str, &str)] = match fs::read_to_string(src).unwrap().contains("LANES") {
            true => &[("", "-DLANES=64"), ("_W32", "-DLANES=32")],
            false => &[("", "")],
        };
        for (suffix, define) in variants {
            let name = format!("{}{suffix}", stem.to_uppercase());
            let spv = out.join(format!("{name}.spv"));
            let status = Command::new(&glslc)
                .args(["--target-env=vulkan1.3", "-O", "-I"])
                .arg(dir)
                .args(Some(define).filter(|d| !d.is_empty()))
                .arg(src)
                .arg("-o")
                .arg(&spv)
                .status()
                .unwrap_or_else(|e| panic!("running {}: {e} (install the Vulkan SDK or set GLSLC)", glslc.display()));
            assert!(status.success(), "glslc failed on {}", src.display());
            consts.push_str(&format!(
                "pub const {name}: &[u8] = include_bytes!({:?});\n",
                spv.display().to_string()
            ));
        }
    }
    fs::write(out.join("shaders.rs"), consts).unwrap();
}
