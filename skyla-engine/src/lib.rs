mod arena;
mod engine;
mod graph;
mod kernels;
mod weights;

pub use engine::Engine;

use anyhow::{anyhow, Result};
use skyla_vk::{Context, MemoryPropertyFlags};

fn host_memory(ctx: &Context) -> Result<u32> {
    let excluded = MemoryPropertyFlags::DEVICE_COHERENT_AMD | MemoryPropertyFlags::DEVICE_UNCACHED_AMD;
    let preferred = MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT;
    ctx.find_memory_type(preferred | MemoryPropertyFlags::HOST_CACHED, excluded)
        .or_else(|| ctx.find_memory_type(preferred, excluded))
        .ok_or_else(|| anyhow!("no host-visible coherent memory type"))
}
