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
| Decode ceiling at 35 GB/s | ~55 tok/s | ~7.0 tok/s |
| Decode ceiling at 42 GB/s (best observed placement) | ~66 tok/s | ~8.4 tok/s |

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
| `skyla-tok` | GGUF byte-level BPE tokenizer (qwen2 pre-tokenizer) |
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
- **M4 — Decode tactics.** Per-format matvec tactics chosen by A/B evidence, faster
  RMSNorm; fusion and device-local staging measured and rejected (done). Still open: a
  `skyla compile` step that runs the sweeps on the target and stores the evidence.
- **M5 — Long reasoning.** Split-K decode attention (done in M2), tokenizer and the R1
  chat template, streaming `skyla chat` with the `<think>` section shown separately, 8-bit
  KV cache with per-tensor precision choice and a 16k default context (done).
- **M6 — Prefill.** Batched matrix-matrix kernels for prompt processing (done).
- **M7 — MoE.** Expert routing and grouped expert matrix-vector for Qwen3-30B-A3B
  (done: decode and batched prefill).

## M7 result: Qwen3-30B-A3B (qwen3moe)

48 layers, 128 experts per layer, 8 routed per token (renormalized softmax over the
chosen 8), expert FFN 768, GQA 32:4. The 17.5 GiB of packed weights load into 9 buffers
in ~30 s; each token touches ~3B parameters.

- The existing Q8_0/Q4_K/Q6_K matrix-vector kernels gained a `MOE` specialization: the
  grid's second dimension is the expert slot and the weight row base is
  `expert_id × n_out`, read from GPU memory, so one dispatch runs all 8 experts.
- Routing is two kernels: router logits with one workgroup per expert, then a parallel
  top-k (subgroup max/min over 256 lanes, 8 rounds) and softmax.
- Expert `down` matrices are only 768 wide (24 Q4_K or 12 Q6_K units per row), which left
  most of a wave idle. A `LPR` (lanes per row) specialization runs 2 or 4 rows per wave
  with a shuffle-xor reduction per lane group; the host picks 64, 32 or 16 per matrix.
- Correctness: greedy output matches llama.cpp's Vulkan backend token for token ("The
  capital of France is Paris." then `<|im_end|>`) and the f32 CPU reference on every
  step; the worst logit difference is ~1.9 with f16 or 8-bit values alike, likely from
  near-tied routing decisions across 48 layers (not yet verified).

| Step | Decode (tok/s) | GPU ms/token | Change |
|---|---:|---:|---|
| First working version | 8.88 | 104.6 | one-workgroup router: 20.7 ms |
| Router logits on 128 workgroups | — | 105.8 | logits 3.8 ms, but serial top-k 19.7 ms |
| Parallel top-k | — | 86.3 | routing ~4 ms total |
| **Narrow-row expert kernels (`LPR`)** | **14.1** | **68.4** | expert down 28.7 → 14.6 ms |

Bugs found by bisecting with a 1-layer model and comparing GPU arena values with the CPU
reference: the router's softmax overwrote its own maximum (`top[0]`) in the first loop
iteration, which left every expert choice right but every weight wrong.

### Batched MoE prefill

A prefill chunk routes every token at once (router logits on an 
_expert × tokens
grid, one top-k workgroup per token), then a single-workgroup sort groups the
(token, slot) pairs by expert and writes a count/offset table plus the sorted list.
The prefill GEMM gained a MOE specialization: the grid's second dimension is the
expert, each workgroup gathers its columns through the sorted list (gate/up read the
token's normalized hidden state, down reads the slot's activation) and scatters its
outputs back by slot. Experts nobody picked exit at once.

With 32-token chunks the expert GEMMs read all ~16 GB of expert weights per chunk at
~36 GB/s, so the chunk grew to 128 tokens; the GEMM now tiles columns in 32-token steps
along the grid's third dimension, which dense models use too.

| Prompt processing, Qwen3-30B-A3B (512-token prompt) | tok/s |
|---|---:|
| One token per decode step | 14.3 |
| Grouped expert GEMM, 32-token chunks | 46.8 |
| **128-token chunks** | **95** |

Dense prefill is unchanged by the larger chunk (Qwen3-4B 116 tok/s, R1-8B 64 tok/s on a
512-token prompt). All four models match the CPU reference with 0 top-1 disagreements
on a 171-token prompt that spans two chunks.

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

## M3 result: DeepSeek-R1-0528-Qwen3-8B on the iGPU

- Q4_K and Q6_K matrix-vector kernels. Q4_K keeps its 16-byte superblock header
  (d, dmin, packed 6-bit scales) in one region and the nibbles in another, so traffic
  stays at 144 bytes per 256 weights; the min term is folded in through Σx. Q6_K blocks
  (210 bytes, not 16-byte aligned) are split into low-nibble, high-bit, scale and d
  regions.
- Weights are planned across several buffers (4455 MiB in 3, under the 2 GiB allocation
  cap) with each matrix kept inside one buffer; each matvec pipeline holds one descriptor
  set per weight buffer.
- Token embedding is dequantized on the host straight from the memory-mapped GGUF into
  the arena, which works for every quantization type.
- YaRN RoPE scaling (factor 4, original context 32768, `mscale = 1 + 0.1·ln 4`) is
  computed once into a per-pair frequency table shared by the CPU reference and the GPU.

Correctness: the CPU reference matches **llama.cpp's Vulkan backend** to 3–4 decimals on
the first-step distribution (" a" 0.2157 vs 0.2156). llama.cpp's own CPU backend
disagrees with its Vulkan backend on this model (0.0858 for the same token), so the
Vulkan backend is the oracle for the 8B model. The GPU engine matches the reference on
12 of 12 greedy tokens, worst logit difference 0.05, validation layer clean.

| R1-0528-Qwen3-8B Q4_K_M, decode | tok/s |
|---|---:|
| llama.cpp b12032, Vulkan | 4.00 |
| llama.cpp b12032, CPU | 4.20 |
| **Skyla (M3)** | **6.80** |

Weight matvecs run at 29–32 GB/s. RMSNorm (one workgroup, 47 µs per call) costs 2.2%.

## M5 result: chat

- `skyla-tok`: byte-level BPE read from GGUF (`gpt2` model, `qwen2` pre-tokenizer via
  `fancy-regex`, control and user-defined tokens matched literally first). It produces
  ids identical to `llama-tokenize` on 8 of 8 test strings covering emoji, CJK, code,
  CRLF, contractions, numbers and a full chat prompt, and every case round-trips.
- `skyla chat`: the DeepSeek-R1 template
  (`<｜begin▁of▁sentence｜>{system}<｜User｜>{msg}<｜Assistant｜>`), temperature / top-k /
  top-p sampling (defaults 0.6 / 40 / 0.95 as DeepSeek recommends for R1), UTF-8-safe
  streaming, the `<think>` section dimmed on a terminal, and the KV cache kept across
  turns. Unlike the Jinja template, earlier turns keep their thinking in context, which
  avoids re-prefilling them at decode speed.
- Example: asked in Indonesian for 17 × 23, the model reasons in Indonesian and answers
  391 at 6.8 tok/s; a second turn recalls facts from the first.

## M6 result: batched prefill

A second recorded graph processes up to 32 prompt tokens per replay; the token count
lives in GPU memory, so one recording serves every chunk size. Norms, QK-norm + RoPE,
attention (causal per token), merge and SiLU take the token as a grid dimension; only
the last token's row reaches the output projection (`take_last`).

The weight GEMM went through three tactics, each checked against the CPU reference:

| Tactic | R1-8B prefill (tok/s) | Why it moved |
|---|---:|---|
| Batched matvec, x re-read per row | 8.3 | ~600 GB of cache traffic for x per batch |
| LDS-tiled GEMM, 128 rows × 16 tokens, f32 math | 27.8 | weights dequantized once per tile into LDS as f16 |
| 128 × 32 tile, 4×4 micro-tile, f32 math | 33–34 | half the LDS reads per FMA |
| **128 × 32 tile, packed-f16 FMA within each 64-wide K step, f32 across steps** | **60–62** | `v_pk_fma_f16`, no per-read conversions |

llama.cpp b12032 Vulkan prefills the same model at 35.6 tok/s. Qwen3-0.6B prefills at
686 tok/s (llama.cpp: 436).

Numerical semantics of the packed-f16 tactic: weights and activations are rounded to
f16 in LDS and 64 products are summed in f16 before joining the f32 accumulator. Over
70 prompt tokens and 16 decode steps on R1-8B, every greedy token matched the f32 CPU
reference with identical probabilities to 4 decimals; the worst logit difference was
0.27, on a token the model predicts with p = 0.9996, and 0.03–0.14 elsewhere.

The attention kernel also stopped writing partial results for context chunks beyond
the current position (previously ~77 MB of dead writes per decode step on the 8B model).

## M4 result: decode tactics, accepted and rejected

Bandwidth on this laptop moves with page placement and temperature, so every decision
below comes from interleaved A/B runs of separately built binaries in one session
(R1-8B, greedy decode, mean of 2–3 runs each), never from runs taken at different times.

| Tactic | Decode (tok/s) | Verdict |
|---|---:|---|
| M6 baseline: 4 waves × 2 rows per wave for every format | 7.03 | |
| Activations in `DEVICE_LOCAL` memory with staging copies | 6.84 | rejected |
| … plus RMSNorm fused into the next matvec | 5.59 | rejected |
| … plus SiLU fused into the down projection | 6.00 | rejected |
| … plus both fusions | 5.00 | rejected |
| **Per-format matvec tactic: K-quants 2 waves × 4 rows, Q8_0 4 × 2** | **7.55** | accepted |
| **… plus RMSNorm with 1024 threads (3.1 ms → 0.43 ms per step)** | **7.58** | accepted |

Fusion loses here because each lane then loads twice the activations (x and the norm
weight, or gate and up) and holds more registers, which lowers occupancy in kernels
that need every resident wave to hide DRAM latency. The staging design sped up
RMSNorm but made other small kernels slower and added copies.

Rows-per-wave sweep for the K-quant kernels (waves 1–8 × rows 1–16): 1 row 5.3–5.6,
2 rows 6.5–7.1, 4 rows 7.5–7.6, 8 rows 7.6, 16 rows 6.3–6.4 tok/s. The best Q8_0
setting on Qwen3-0.6B was 2 rows, so the tactic is chosen per weight format.

After M4 the weight matvecs read at 36–38 GB/s against a ~40 GB/s roofline measured in
the same session; norms, RoPE, attention and SiLU together cost ~3% of a decode step.

## M5 result: 8-bit KV cache

Keys and values can each be stored as f16 or as int8 with one f16 scale per 32 elements
(Q8_0-style, 1088 bytes per head per position instead of 2048). The QK-norm/RoPE kernel
quantizes in registers; attention dequantizes on load. `--kv` picks the precision:

| `--kv` | Keys | Values | R1-8B worst logit error* | Decode at 4k context | KV for 16k context |
|---|---|---|---:|---:|---:|
| `f16` | f16 | f16 | 0.15 | 6.57 tok/s | 2304 MiB (over the 2 GiB cap) |
| **`v8` (default)** | f16 | int8 | **0.19** | 6.72 tok/s | **1.84 GiB** |
| `q8` | int8 | int8 | 2.01 | 6.88 tok/s | 1.28 GiB |

\* Against the f32 CPU reference over 70 prompt tokens and 16 decode steps; every
greedy token matched for all three. Attention at 4k context took 20.6 / 16.1 / 13.4 ms
per step. Qwen3 keys carry large per-dimension outliers after QK-norm (values near
200), so one outlier forces a coarse step on the rest of its 32-element block; values
have no such outliers, which is why `v8` is the default and `skyla chat` now opens a
16k context.

Driver finding: `subgroupClusteredMax(x, 32)` on this AMD driver (23.19) returns wrong
results for the upper half of a 64-lane wave, which zeroed outlier keys. The block
maximum is now an explicit `subgroupShuffleXor` butterfly over offsets 1–16.

Prefill slows with context (39.7 tok/s at 4k vs 60 at short prompts) because each of
the batched queries reads the whole KV cache separately; sharing KV reads across
the batch is the next prefill improvement.

## Port: Apple M5 through MoltenVK

Apple GPUs run 32-lane SIMD groups, and every subgroup reduction here was written for
wave64: on the M5 each `subgroupAdd` summed only half of a row, and the first run
disagreed with the reference on every token (worst logit error 26). Kernels that reduce
across a subgroup (`matvec_*`, `attention`, `qkv_prep`, `rmsnorm`) now take a `LANES`
macro; `build.rs` compiles them twice, `NAME` with 64 lanes (the Vega code) and
`NAME_W32` with 32, and the engine picks by the device's subgroup size. With 32 lanes,
attention holds 4 head dimensions per lane instead of 2 and QK-norm/RoPE handles two
rotation pairs per lane. Other host changes: the instance enables
`VK_KHR_portability_enumeration`, the device enables `VK_KHR_portability_subset`, weight
buffers are capped at `maxStorageBufferRange` (4 GiB on MoltenVK, below its 8.9 GiB
allocation limit), and `time_dispatch` uses consecutive bottom-of-pipe timestamps,
because MoltenVK cannot timestamp around a single dispatch (`bench-bw` reported 6500 GB/s).

| Apple M5, MoltenVK 1.4 | Worst logit error (`f16` / `v8` / `q8`) | Decode | Prefill (473 tok) |
|---|---|---:|---:|
| Qwen3-0.6B Q8_0 | 0.03 / 0.13 / 1.61 | 117 tok/s | 1098 tok/s |
| Qwen3-0.6B Q4_K_M | 0.05 / 0.17 / 1.59 | 122 tok/s | 1281 tok/s |
| R1-8B Q4_K_M (2 weight buffers) | – / 0.13 / – | 19–24 tok/s | 100–114 tok/s |

Errors are against the CPU reference over 24 steps (R1-8B: 16), all with zero top-1
disagreements; a 473-token prompt over 80 steps ran clean under the validation layer.
GPU read bandwidth is 134 GB/s. On R1-8B the Q4_K gate+up matvec reads at ~87 GB/s and
the Q6_K output head at ~108 GB/s. A sweep of K-quant tactics (waves × rows) gave
17–24 tok/s with run-to-run noise larger than the differences, so the Vega tactics stay
until a controlled A/B says otherwise.

## Native Metal backend (Apple M5)

MoltenVK translates every Vulkan submit into Metal at run time and cannot reach the
GPU's matrix units (no `VK_KHR_cooperative_matrix`). On macOS the engine now runs on
`skyla-metal` instead; `--features vulkan` keeps the MoltenVK path for A/B runs and
every other platform is unchanged.

- `skyla-metal` mirrors the `skyla-vk` surface the engine uses (context, shared
  buffers, pipelines with several binding sets, recorded and replayed command lists,
  per-stage GPU times), so the engine only swaps an import.
- Every GLSL kernel has an MSL twin in `skyla-metal/kernels`
  ([conventions](../skyla-metal/kernels/CONVENTIONS.md)). Specialization constants
  become `SPEC_n` macros and each (kernel, spec) pair is compiled from source once,
  so arrays sized by a specialization stay compile-time.
- A recording is replayed into one concurrent compute encoder with a buffer memory
  barrier wherever the Vulkan graph has one; encoding 831 dispatches for an R1-8B
  token takes ~0.15 ms.
- The prefill GEMM keeps the K-quant dequantization into threadgroup memory but
  multiplies with `simdgroup_matrix` (8 SIMD groups, 16 rows × 32 tokens each, f32
  accumulators).
- Weights are read with `F_NOCACHE` into the GPU buffers instead of being copied from
  the memory map, so a model is not held twice (MoE REAP-15B load: 14 s → 4.2 s).
- K-quant decode uses 8 SIMD groups × 1 row on 32-lane GPUs (GPU time per R1-8B token
  −8% in ABBA runs); Vega keeps 2 × 4.

All five test models match the CPU reference with 0 top-1 disagreements.

| Apple M5, 512-token prompt + 128 tokens, warm | MoltenVK | Metal | MLX 0.32 |
|---|---:|---:|---:|
| Qwen3-0.6B Q8_0, decode / prefill (tok/s) | 104 / 1 780 | **155 / 2 280** | 164 / 7 517 |
| Qwen3-4B Q4_K_M | 41.8 / 324 | **47.5 / 372** | 50.7 / 1 319 |
| R1-8B Q4_K_M | 24.0 / 178 | **26.2 / 203** | 28.2 / 723 |

Decode is within 6–7% of MLX, roughly the size difference between Q4_K_M (Q6_K output
head and part of `down`) and MLX's uniform 4.5-bit weights; R1-8B decode reads at ~127
GB/s against a measured 134 GB/s. Prefill is the open gap.

Measurement notes: run-to-run results on this laptop drift by up to 30% once it is
hot (and while it charges), so A/B results come from ABBA-ordered runs after a
cool-down, and `skyla run --warmup` runs the prompt once first, as
`mlx_lm.benchmark` does. On a 16 GB machine the first prompt after loading a 9 GB
model can still take seconds while macOS decompresses swapped weight pages.

### Finding: memory placement decides bandwidth

The laptop pairs an 8 GB and a 32 GB module: the first 16 GB run dual-channel and the
remaining 24 GB single-channel. Where Windows places a buffer's pages is outside Skyla's
control, so measured bandwidth moves between runs. After M3, the same Q8_0 kernels that
gave Qwen3-0.6B 41 tok/s gave 55 tok/s with the output projection reading at 42.6 GB/s,
above the 35 GB/s measured in M0. The M0 roofline is therefore a placement-dependent
lower bound, not a ceiling.
