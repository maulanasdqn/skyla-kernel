use crate::probe::flags_str;
use anyhow::Result;
use skyla_vk::{BufferUsageFlags, Context, MemoryPropertyFlags};
use std::hint::black_box;
use std::time::Instant;

const WG_SIZES: [u32; 3] = [64, 128, 256];
const GROUP_COUNTS: [u32; 5] = [64, 256, 1024, 4096, 16384];

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

pub fn bandwidth(size: u64, reps: u32) -> Result<()> {
    let ctx = Context::new()?;
    println!("device: {}\n", ctx.device_name());
    gpu(&ctx, size, reps)?;
    println!();
    cpu(size, reps);
    Ok(())
}

fn gpu(ctx: &Context, size: u64, reps: u32) -> Result<()> {
    let pipelines = WG_SIZES
        .iter()
        .map(|&wg| {
            ctx.create_pipeline(skyla_shaders::BANDWIDTH, 2, 4, &[wg])
                .map(|p| (wg, p))
        })
        .collect::<Result<Vec<_>>>()?;
    let max_threads = (WG_SIZES[2] * GROUP_COUNTS[4]) as u64;
    let dst_type = ctx
        .find_memory_type(MemoryPropertyFlags::HOST_VISIBLE, MemoryPropertyFlags::empty())
        .expect("host-visible memory");
    let dst = ctx.alloc_buffer(max_threads * 4, BufferUsageFlags::STORAGE_BUFFER, dst_type)?;
    let unusable = MemoryPropertyFlags::LAZILY_ALLOCATED
        | MemoryPropertyFlags::PROTECTED
        | MemoryPropertyFlags::DEVICE_COHERENT_AMD
        | MemoryPropertyFlags::DEVICE_UNCACHED_AMD;

    println!("GPU streaming read, {} MiB per pass", size >> 20);
    println!(
        "  {:<4} {:<38} {:>9}  {:>6} {:>7}",
        "type", "flags", "GB/s", "wg", "groups"
    );
    for (ti, t) in ctx.memory_types().iter().enumerate() {
        let flags = t.property_flags;
        if flags.is_empty() || flags.intersects(unusable) {
            continue;
        }
        let heap = ctx.memory_heaps()[t.heap_index as usize].size;
        let bytes = size.min(heap / 2).min(ctx.max_allocation_size) & !0xffff;
        let src = match ctx.alloc_buffer(bytes, BufferUsageFlags::STORAGE_BUFFER, ti as u32) {
            Ok(b) => b,
            Err(e) => {
                println!("  {ti:<4} {:<38} skipped: {e}", flags_str(flags));
                continue;
            }
        };

        let push = ((bytes / 16) as u32).to_le_bytes();
        let mut best = (0.0f64, 0u32, 0u32);
        for (wg, pipe) in &pipelines {
            pipe.bind_buffers(&[&src, &dst]);
            for &groups in &GROUP_COUNTS {
                ctx.time_dispatch(pipe, [groups, 1, 1], &push, 1)?;
                let ns = median(ctx.time_dispatch(pipe, [groups, 1, 1], &push, reps)?);
                let gbs = bytes as f64 / ns;
                if gbs > best.0 {
                    best = (gbs, *wg, groups);
                }
            }
        }
        let note = match bytes < size {
            true => format!("  ({} MiB, heap-limited)", bytes >> 20),
            false => String::new(),
        };
        println!(
            "  {ti:<4} {:<38} {:>9.2}  {:>6} {:>7}{note}",
            flags_str(flags),
            best.0,
            best.1,
            best.2
        );
    }
    Ok(())
}

fn cpu(size: u64, reps: u32) {
    let words = (size / 8) as usize;
    let data: Vec<u64> = (0..words as u64).collect();
    let max_threads = std::thread::available_parallelism().map_or(8, |n| n.get());

    println!("CPU streaming read, {} MiB per pass", size >> 20);
    let mut threads = 1;
    while threads <= max_threads {
        let chunk = words.div_ceil(threads);
        let times: Vec<f64> = (0..reps)
            .map(|_| {
                let t = Instant::now();
                std::thread::scope(|s| {
                    for part in data.chunks(chunk) {
                        s.spawn(move || {
                            let mut acc = [0u64; 8];
                            for w in part.as_chunks::<8>().0 {
                                for i in 0..8 {
                                    acc[i] ^= w[i];
                                }
                            }
                            black_box(acc);
                        });
                    }
                });
                t.elapsed().as_nanos() as f64
            })
            .collect();
        println!("  {threads:>2} threads  {:>7.2} GB/s", size as f64 / median(times));
        threads *= 2;
    }
}
