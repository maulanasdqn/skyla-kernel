pub struct Shader {
    pub name: &'static str,
    pub spirv: &'static [u8],
}

include!(concat!(env!("OUT_DIR"), "/shaders.rs"));
