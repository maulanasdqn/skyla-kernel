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
| **Target model** | DeepSeek-R1-0528-Qwen3-8B, abliterated (Josiefied v1), Q4_K_M, batch 1 |
| Development model | Qwen3-0.6B, Q8_0: same `qwen3` architecture, 13× smaller, for fast correctness loops |
| Stretch model | Qwen3-30B-A3B (MoE), Q4 |

### Target model shape

DeepSeek-R1-0528-Qwen3-8B is a Qwen3-8B fine-tune, so every kernel validated on
Qwen3-0.6B carries over with different constants:

| | Qwen3-0.6B | R1-0528-Qwen3-8B |
|---|---:|---:|
| Layers | 28 | 36 |
| Hidden size | 1024 | 4096 |
| Query / KV heads | 16 / 8 | 32 / 8 |
| Head dim | 128 | 128 |
| FFN size | 3072 | 12288 |
| Vocabulary | 151 936 | 151 936 |
| Weights per token | 633 MB (Q8_0) | 5.03 GB / 4.68 GiB (Q4_K_M: Q4_K + Q6_K blocks) |
| Decode ceiling at 35 GB/s | ~55 tok/s | **~7.0 tok/s** |

R1 is a reasoning model and routinely writes thousands of tokens before answering, so
long-context decode matters. An FP16 KV cache costs 144 KiB per token (36 layers × 2 ×
8 heads × 128 × 2 bytes): at 8k tokens every decode step reads 1.2 GB of cache on top of
the weights, about 25% more traffic. Attention must therefore be a split-K
("flash-decoding") kernel, and a Q8 KV cache is a planned tactic.

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
| `skyla-gguf` | Memory-mapped GGUF reader with F32, F16, Q8_0, Q4_K and Q6_K dequantization |
| `skyla-ref` | Plain CPU forward pass, the correctness oracle |
| `skyla-engine` | Packed weights, static arena, recorded decode graph, `Engine::step` |
| `skyla-compiler` (M4) | Contracts, tactic selection, layout and memory planning |

All crates sit at the repository root. Code carries no comments and no source file
exceeds 200 lines.

## Milestones

- **M0 — Roofline.** Device probe, GPU read bandwidth per memory type, CPU bandwidth.
  Baseline: llama.cpp b12032 Vulkan on the same model.
- **M1 — Reference.** GGUF loader and a CPU reference forward pass for Qwen3-0.6B.
  Acceptance: teacher-forced on llama.cpp's greedy output, our top-1 matches at every step
  except near-ties (top-2 margin under 2 points), since llama.cpp's CPU path quantizes
  activations to 8 bits and Skyla's reference stays in f32.
- **M2 — GPU decode.** Kernels for Q8_0 matrix-vector, RMSNorm, RoPE, QK-norm, decode
  attention, SwiGLU and residual add. Full decode on the iGPU, checked layer by layer
  against `skyla-ref`.
- **M3 — K-quants.** Q4_K and Q6_K matrix-vector kernels, checked against the CPU
  reference. Load R1-0528-Qwen3-8B, splitting weights across buffers (2 GiB allocation cap).
- **M4 — Ahead-of-time engine.** `skyla compile` emits an engine: autotuned tactics with
  evidence, repacked weights, a static memory plan and a pre-recorded command buffer.
  Goal: beat llama.cpp decode speed on R1-0528-Qwen3-8B.
- **M5 — Long reasoning.** Split-K decode attention, Q8 KV cache, tokenizer and the R1
  chat template, streaming `skyla chat` with the `<think>` section shown separately.
- **M6 — Prefill.** Batched matrix-matrix kernels for prompt processing.
- **M7 — MoE.** Expert routing and grouped expert matrix-vector for Qwen3-30B-A3B.

## Baseline (llama.cpp b12032, Qwen3-0.6B Q8_0)

| Backend | pp512 (tok/s) | tg128 (tok/s) |
|---|---:|---:|
| Vulkan, all layers on iGPU | 435.58 ± 11.17 | 44.53 ± 0.72 |
| CPU | 378.56 ± 13.31 | 38.51 ± 2.34 |

At 44.5 tok/s the iGPU decode streams about 27 GB/s of weights. The theoretical peak
for DDR4-3200 dual-channel is 51.2 GB/s.

## Baseline (llama.cpp b12032, R1-0528-Qwen3-8B abliterated Q4_K_M)

| Backend | pp512 (tok/s) | tg128 (tok/s) |
|---|---:|---:|
| Vulkan, all layers on iGPU | 35.64 ± 1.29 | 4.00 ± 0.37 |
| CPU | 34.03 ± 0.49 | 4.20 ± 0.09 |

On the target model llama.cpp's iGPU decode is slower than its CPU decode and streams
only about 20 GB/s, 57% of the measured 35 GB/s roofline. The ceiling is about
7.0 tok/s, so a Vega-specific Q4_K/Q6_K path has up to 1.75× of headroom.

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
  (633 MB of weights per token) is about 55 tok/s.

## M1 result: reference agrees with llama.cpp

Prompt "The capital of France is" (ids `785,6722,315,9625,374`), 24 tokens of llama.cpp
b12032 CPU greedy output teacher-forced through `skyla ref --expect`:

- First-step probability of " Paris": 0.6490 (Skyla) vs 0.6517 (llama.cpp).
- 23 of 24 steps agree on top-1. The one disagreement is a near-tie: Skyla gives
  "France" 0.1799 and "Italy" 0.1680, llama.cpp picks "Italy".
- The f32 reference decodes at 3.4 tok/s on the CPU; it is an oracle, not an engine.

## M2 result: full decode on the iGPU

`skyla-engine` packs Q8_0 weights into a kernel-native layout (int8 quants and f16
scales in separate, 16-byte aligned regions of one buffer), plans every activation at
a fixed offset in one arena, and records the whole decode step into a single command
buffer that is replayed per token; the token and position live in GPU memory.

Kernels: multi-row Q8_0 matrix-vector (wave64, one `uvec4` of weights per lane per
step), RMSNorm, fused QK-norm + RoPE + KV-cache write, split-K decode attention with
online softmax and GQA sharing, attention merge, SiLU·mul, Q8_0 embedding.

Correctness (`skyla run --check`, Khronos validation layer on): 80 of 80 greedy tokens
match the CPU reference, worst logit difference 0.024 (f16 KV cache).

Speed on Qwen3-0.6B Q8_0 (`skyla run -n 129`): **41.3 tok/s** vs llama.cpp 44.5 tok/s.
`skyla run --profile` for one decode step at position ~133 (23.3 ms):

| Stage | Share | Notes |
|---|---:|---|
| Matrix-vector, per layer | ~70% | 28–31 GB/s; each dispatch reads only 2–7 MB, so ramp-up and tail cost ~10% |
| Output projection | ~21% | 34.6 GB/s, at the roofline |
| Attention + merge | ~6% | 27 GB/s of KV reads at position 1500 after the split-K rewrite (was 2.3 ms at position 133) |
| Norms, RoPE, SiLU, embedding | ~2% | |

Tactic sweeps: rows per workgroup (1–16) made no difference for the one-row-per-wave
kernel; the multi-row kernel's best setting is 4 waves × 2 rows (+2%). The remaining gap
on this small model is per-dispatch ramp-up. The 8B target's matrices are 4–16× larger,
which amortizes it; kernel fusion (norm into matvec, SiLU into the down projection) is
planned for M4.
