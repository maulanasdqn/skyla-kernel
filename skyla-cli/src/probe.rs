use anyhow::Result;
use skyla_vk::{Context, MemoryPropertyFlags};

const FLAG_NAMES: [(MemoryPropertyFlags, &str); 8] = [
    (MemoryPropertyFlags::DEVICE_LOCAL, "DEVICE_LOCAL"),
    (MemoryPropertyFlags::HOST_VISIBLE, "HOST_VISIBLE"),
    (MemoryPropertyFlags::HOST_COHERENT, "HOST_COHERENT"),
    (MemoryPropertyFlags::HOST_CACHED, "HOST_CACHED"),
    (MemoryPropertyFlags::LAZILY_ALLOCATED, "LAZY"),
    (MemoryPropertyFlags::PROTECTED, "PROTECTED"),
    (MemoryPropertyFlags::DEVICE_COHERENT_AMD, "DEV_COHERENT_AMD"),
    (MemoryPropertyFlags::DEVICE_UNCACHED_AMD, "DEV_UNCACHED_AMD"),
];

pub fn flags_str(f: MemoryPropertyFlags) -> String {
    let mut parts: Vec<String> = FLAG_NAMES
        .iter()
        .filter(|(b, _)| f.contains(*b))
        .map(|(_, n)| n.to_string())
        .collect();
    let known = FLAG_NAMES
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

pub fn run() -> Result<()> {
    let ctx = Context::new()?;
    let p = &ctx.props;
    let l = &p.limits;
    let v = |x: u32| format!("{}.{}.{}", x >> 22, (x >> 12) & 0x3ff, x & 0xfff);

    println!("device          {} ({:?})", ctx.device_name(), p.device_type);
    println!("vendor/device   {:#06x}/{:#06x}", p.vendor_id, p.device_id);
    println!("api / driver    {} / {:#x}", v(p.api_version), p.driver_version);
    println!("subgroup size   {}", ctx.subgroup_size);
    println!("caps            {:?}", ctx.caps);
    println!(
        "max wg size     {} ({:?})",
        l.max_compute_work_group_invocations, l.max_compute_work_group_size
    );
    println!("max wg count    {:?}", l.max_compute_work_group_count);
    println!("shared memory   {} KiB", l.max_compute_shared_memory_size / 1024);
    println!("max ssbo range  {} MiB", l.max_storage_buffer_range as u64 >> 20);
    println!("max allocation  {} MiB", ctx.max_allocation_size >> 20);
    println!("timestamp       {} ns/tick", l.timestamp_period);

    println!("\nheaps");
    for (i, h) in ctx.memory_heaps().iter().enumerate() {
        println!("  {i}  {:>7} MiB  {:?}", h.size >> 20, h.flags);
    }
    println!("memory types");
    for (i, t) in ctx.memory_types().iter().enumerate() {
        println!("  {i}  heap {}  {}", t.heap_index, flags_str(t.property_flags));
    }
    Ok(())
}
