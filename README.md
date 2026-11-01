# Skyla Kernel

**Ahead-of-time LLM inference kernels for one exact machine: an AMD Vega 8 iGPU.**

Skyla takes the idea behind [Netra Kernel](https://github.com/NetraRuntime/netra-kernel)
(compile for the exact serving contract and launch only specialized kernels) and points
it at a consumer laptop instead of a datacenter GPU. ROCm does not support this GPU, so
Skyla's kernels are Vulkan compute shaders, specialized at pipeline-creation time and
tuned by measurement on the target device.

> Status: **M0 done (roofline)**, M1 in progress. Nothing runs a model yet.
> See [docs/roadmap.md](docs/roadmap.md).

Measured on the target: the iGPU streams weights at about 35 GB/s, the CPU at about
31 GB/s, and llama.cpp's Vulkan decode reaches about 27 GB/s. Skyla's first goal is to
close that gap.

## Requirements

- Windows 11 with an AMD Vulkan 1.3 driver
- Rust (stable, MSVC toolchain)
- [Vulkan SDK](https://vulkan.lunarg.com/) for `glslc` and the validation layer

## Usage

```powershell
cargo build --release
./target/release/skyla probe      # device, features, limits, memory types
./target/release/skyla bench-bw   # read bandwidth per memory type, GPU and CPU
```

Set `SKYLA_VALIDATION=1` to enable the Khronos validation layer, and `SKYLA_DEVICE=<n>`
to choose a device other than the first integrated GPU.

## License

[MIT](LICENSE)
