mod bench;

use anyhow::Result;
use clap::{Parser, Subcommand};
use skyla_vk::{Context, MemoryPropertyFlags};

#[derive(Parser)]
#[command(
    name = "skyla",
    version,
    about = "Ahead-of-time LLM kernels for one AMD Vega iGPU"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Print the selected Vulkan device, its features, limits and memory types
    Probe,
    /// Measure streaming-read bandwidth of every GPU memory type and of the CPU
    BenchBw {
        /// Bytes read per measurement, in MiB
        #[arg(long, default_value_t = 512)]
        size_mib: u64,
        /// Timed repetitions per configuration (median is reported)
        #[arg(long, default_value_t = 7)]
        reps: u32,
    },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Probe => probe(),
        Command::BenchBw { size_mib, reps } => bench::bandwidth(size_mib << 20, reps),
    }
}

pub fn flags_str(f: MemoryPropertyFlags) -> String {
    let names = [
        (MemoryPropertyFlags::DEVICE_LOCAL, "DEVICE_LOCAL"),
        (MemoryPropertyFlags::HOST_VISIBLE, "HOST_VISIBLE"),
        (MemoryPropertyFlags::HOST_COHERENT, "HOST_COHERENT"),
        (MemoryPropertyFlags::HOST_CACHED, "HOST_CACHED"),
        (MemoryPropertyFlags::LAZILY_ALLOCATED, "LAZY"),
        (MemoryPropertyFlags::PROTECTED, "PROTECTED"),
        (MemoryPropertyFlags::DEVICE_COHERENT_AMD, "DEV_COHERENT_AMD"),
        (MemoryPropertyFlags::DEVICE_UNCACHED_AMD, "DEV_UNCACHED_AMD"),
    ];
    let mut parts: Vec<String> = names
        .iter()
        .filter(|(b, _)| f.contains(*b))
        .map(|(_, n)| n.to_string())
        .collect();
    let known = names
        .iter()
        .fold(MemoryPropertyFlags::empty(), |acc, (b, _)| acc | *b);
    if !(f & !known).is_empty() {
        parts.push(format!("{:#x}", (f & !known).as_raw()));
    }
    if parts.is_empty() {
        "-".into()
    } else {
        parts.join("|")
    }
}

fn probe() -> Result<()> {
    let ctx = Context::new()?;
    let p = &ctx.props;
    let l = &p.limits;
    let v = |x: u32| format!("{}.{}.{}", x >> 22, (x >> 12) & 0x3ff, x & 0xfff);

    println!(
        "device          {} ({:?})",
        ctx.device_name(),
        p.device_type
    );
    println!("vendor/device   {:#06x}/{:#06x}", p.vendor_id, p.device_id);
    println!(
        "api / driver    {} / {:#x}",
        v(p.api_version),
        p.driver_version
    );
    println!("subgroup size   {}", ctx.subgroup_size);
    println!("caps            {:?}", ctx.caps);
    println!(
        "max wg size     {} ({:?})",
        l.max_compute_work_group_invocations, l.max_compute_work_group_size
    );
    println!("max wg count    {:?}", l.max_compute_work_group_count);
    println!(
        "shared memory   {} KiB",
        l.max_compute_shared_memory_size / 1024
    );
    println!(
        "max ssbo range  {} MiB",
        l.max_storage_buffer_range as u64 >> 20
    );
    println!("max allocation  {} MiB", ctx.max_allocation_size >> 20);
    println!("timestamp       {} ns/tick", l.timestamp_period);

    println!("\nheaps");
    for (i, h) in ctx.memory_heaps().iter().enumerate() {
        println!("  {i}  {:>7} MiB  {:?}", h.size >> 20, h.flags);
    }
    println!("memory types");
    for (i, t) in ctx.memory_types().iter().enumerate() {
        println!(
            "  {i}  heap {}  {}",
            t.heap_index,
            flags_str(t.property_flags)
        );
    }
    Ok(())
}
