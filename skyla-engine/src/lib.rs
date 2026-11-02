mod arena;
mod engine;
mod graph;
mod kernels;
mod kv;
mod planner;
mod weights;

pub use engine::Engine;
pub use kv::KvPrecision;

#[cfg(any(not(target_os = "macos"), feature = "vulkan"))]
use skyla_vk as gpu;

#[cfg(all(target_os = "macos", not(feature = "vulkan")))]
use skyla_metal as gpu;
