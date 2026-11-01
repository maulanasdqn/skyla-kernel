# Skyla Kernel roadmap

Skyla applies the [Netra Kernel](https://github.com/NetraRuntime/netra-kernel) idea —
compile for the exact serving contract, launch only specialized kernels — to a single
consumer machine instead of a datacenter GPU.

## Target contract

| | |
|---|---|
| Machine | ASUS Vivobook M1405YA |
| CPU | Ryzen 7 7730U (Zen 3, 8C/16T, AVX2) |
| GPU | Radeon Vega 8 iGPU (`gfx90c`, 8 CU, wave64, 32 KiB LDS) |
| API | Vulkan 1.3, AMD proprietary driver (ROCm does not support this GPU) |
| Memory | 40 GiB DDR4-3200 shared by CPU and GPU (8 + 32 GiB, partially dual-channel) |
| Device features | FP16 arithmetic, 8/16-bit storage. No accelerated int8 dot product, no matrix cores |
| First model | Qwen3-0.6B, Q8_0, batch 1 |
| Stretch model | Qwen3-30B-A3B (MoE), Q4 |

Consequences for kernel design:

- Decode is bound by memory bandwidth, and the GPU and CPU share the same DRAM. Every
  kernel is judged against the measured roofline from `skyla bench-bw`, not peak FLOPS.
- With no int8 dot product, quantized weights are dequantized to FP16/FP32 in registers.
- wave64 is fixed, so reductions are written for 64 lanes and skip size negotiation.

## Architecture

```text
GGUF checkpoint
      │
      ▼
skyla compile        contract (model dims, quant, max context, batch)
      │              tactics chosen by measured benchmarks, stored as evidence
      │              weights repacked into kernel-native layouts
      │              static memory plan (one arena, fixed offsets)
      ▼
Skyla engine dir     manifest.json · weights.bin · tactics.json · evidence/
      │
      ▼
skyla run            load once, record one command buffer per step,
                     replay it per token (position lives in a GPU buffer)
```

Crates:

| Crate | Role |
|---|---|
| `skyla-vk` | Minimal Vulkan layer: device, buffers, specialized pipelines, timestamps |
| `skyla-shaders` | GLSL kernels compiled to SPIR-V at build time |
| `skyla-cli` | `skyla` binary: probe, benchmarks, compile, run |
| `skyla-gguf` (M1) | Memory-mapped GGUF reader |
| `skyla-ref` (M1) | Plain CPU forward pass, the correctness oracle |
| `skyla-compiler` (M3) | Contracts, tactic selection, layout and memory planning |

## Milestones

- **M0 — Roofline.** Device probe, GPU read bandwidth per memory type, CPU bandwidth.
  Baseline: llama.cpp b12032 Vulkan on the same model.
- **M1 — Reference.** GGUF loader and a CPU reference forward pass for Qwen3-0.6B whose
  greedy output matches llama.cpp token for token.
- **M2 — GPU decode.** Kernels for Q8_0 matrix-vector, RMSNorm, RoPE, QK-norm, decode
  attention, SwiGLU and residual add. Full decode on the iGPU, checked layer by layer
  against `skyla-ref`.
- **M3 — Ahead-of-time engine.** `skyla compile` emits an engine: autotuned tactics with
  evidence, repacked weights, a static memory plan and a pre-recorded command buffer.
  Goal: beat llama.cpp decode speed on Qwen3-0.6B.
- **M4 — Prefill.** Batched matrix-matrix kernels for prompt processing.
- **M5 — Q4 weights.** Q4_0 / Q4_K dequantizing kernels.
- **M6 — MoE.** Expert routing and grouped expert matrix-vector for Qwen3-30B-A3B,
  with weights placed by measured bandwidth across memory types.

## Baseline (llama.cpp b12032, Qwen3-0.6B Q8_0)

| Backend | pp512 (tok/s) | tg128 (tok/s) |
|---|---:|---:|
| Vulkan, all layers on iGPU | 435.58 ± 11.17 | 44.53 ± 0.72 |
| CPU | 378.56 ± 13.31 | 38.51 ± 2.34 |

At 44.5 tok/s the iGPU decode streams about 27 GB/s of weights. The theoretical peak
for DDR4-3200 dual-channel is 51.2 GB/s.

## M0 result: measured roofline

`skyla bench-bw --reps 9`, 512 MiB streaming read, best tactic per memory type, two runs:

| Memory type | Heap | GPU read (GB/s) |
|---|---|---:|
| `DEVICE_LOCAL` | 0 (256 MiB carve-out) | 28.9 – 33.0 |
| `HOST_VISIBLE \| HOST_COHERENT` | 1 (system RAM) | 31.9 – 34.9 |
| `DEVICE_LOCAL \| HOST_VISIBLE \| HOST_COHERENT` | 2 (256 MiB) | 29.5 – 31.4 |
| `HOST_VISIBLE \| HOST_COHERENT \| HOST_CACHED` | 1 (system RAM) | **34.8** |

CPU streaming read peaks at 30.7 GB/s with 4 threads and drops with more.

Findings:

- The 256 MiB device-local carve-out is not faster: it is the same DRAM. Weights live in
  system memory (heap 1) at full speed, so the carve-out size does not limit model size.
  Single allocations are capped at 2 GiB, so larger models are split across buffers.
- The GPU reads about 13% faster than the CPU, so the iGPU is the right decode device.
- llama.cpp decode runs at ~77% of the GPU roofline. The ceiling for Qwen3-0.6B Q8_0
  (633 MB of weights per token) is about 55 tok/s; the M3 target is above 50 tok/s.
