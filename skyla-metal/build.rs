use std::path::{Path, PathBuf};
use std::{env, fs};

fn main() {
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let dir = Path::new("kernels");
    println!("cargo:rerun-if-changed=kernels");
    let mut kernels: Vec<_> = fs::read_dir(dir)
        .expect("reading kernels/")
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "metal"))
        .collect();
    kernels.sort();
    let mut table = String::from("pub fn source(name: &str) -> Option<&'static str> {\n    match name {\n");
    for k in &kernels {
        let stem = k.file_stem().unwrap().to_str().unwrap();
        let path = fs::canonicalize(k).unwrap();
        table.push_str(&format!(
            "        {stem:?} => Some(include_str!({:?})),\n",
            path.display().to_string()
        ));
    }
    table.push_str("        _ => None,\n    }\n}\n");
    fs::write(out.join("kernels.rs"), table).unwrap();
}
