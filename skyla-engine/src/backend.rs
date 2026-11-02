#[cfg(any(not(target_os = "macos"), feature = "vulkan"))]
mod selected {
    pub use skyla_vk as gpu;
    pub const MATMUL_TILE_COLS: u32 = 32;
    pub const MOE_TILE_COLS: u32 = 32;
    pub const HALF_INPUTS: bool = false;
}

#[cfg(all(target_os = "macos", not(feature = "vulkan")))]
mod selected {
    pub use skyla_metal as gpu;
    pub const MATMUL_TILE_COLS: u32 = 64;
    pub const MOE_TILE_COLS: u32 = 32;
    pub const HALF_INPUTS: bool = true;
}

pub use selected::*;

pub const TO_HALF: skyla_shaders::Shader = skyla_shaders::Shader {
    name: "to_half",
    spirv: &[],
};
