# Skyla Kernel

**Ahead-of-time LLM inference kernels for laptop GPUs: an AMD Vega 8 iGPU and the Apple M5.**

Skyla runs DeepSeek-R1-0528-Qwen3-8B on a Ryzen 7 7730U laptop with no discrete GPU,
**1.9× faster than llama.cpp at generation and 1.7× faster at prompt processing** on the
same hardware, and checks every kernel against a plain CPU reference. On an Apple M5
MacBook it runs on a native Metal backend that multiplies on the GPU's neural accelerators:
**26.6 tok/s generation and 509 tok/s prompt processing** on the same 8B model, within 9% of
MLX at generation. It ships a terminal chat and a browser chat UI (Rust/WebAssembly) that
can switch between models at runtime.

It applies the idea behind [Netra Kernel](https://github.com/NetraRuntime/netra-kernel),
compile for the exact serving contract and launch only specialized kernels, to a consumer
laptop instead of a datacenter GPU. ROCm does not support the Vega, so its kernels are
Vulkan compute shaders; on Apple silicon every kernel has a Metal twin. Both are specialized
at pipeline creation and chosen by measurement on the target device.

## Results

### ASUS Vivobook (AMD Vega 8)

DeepSeek-R1-0528-Qwen3-8B (Josiefied abliterated v1, Q4_K_M, 4.68 GiB), batch size 1,
Windows 11, AMD driver 23.19, against llama.cpp b12032 on the same laptop:

| | Skyla | llama.cpp Vulkan | llama.cpp CPU |
|---|---:|---:|---:|
| Generation (tok/s) | **7.6** | 4.0 | 4.2 |
| Prompt processing (tok/s) | **60** | 35.6 | 34.0 |
| Default context | 16 384 tokens | | |

Other models on the same laptop (Skyla, 256-token prompt):

| Model | Size | Generation | Prompt processing | Notes |
|---|---:|---:|---:|---|
| Qwen3-30B-A3B-Instruct-2507 (Q4_K_M, MoE) | 17.3 GB | 14.1 tok/s | 95 tok/s | 128 experts, 8 active; best quality per token on this laptop |
| Qwen3-4B-Instruct-2507 (Q4_K_M) | 2.3 GB | 12.0 tok/s | 118 tok/s | No thinking; good for fast role-play and game characters |
| DeepSeek-R1-0528-Qwen3-8B (Q4_K_M) | 4.7 GB | 7.6 tok/s | 60 tok/s | Reasoning model; writes a `<think>` section first |
| Qwen3-0.6B (Q8_0) | 0.6 GB | 55 tok/s | 686 tok/s | Development model (llama.cpp: 44.5 / 436) |

Generation is bound by memory bandwidth. The weight kernels read at 36–38 GB/s against a
measured ~40 GB/s roofline, so Skyla runs at about 95% of what this memory system can
deliver for a 5 GB model.

### Apple M5

macOS 27 on an Apple M5 (10-core GPU, 16 GB unified memory), native Metal backend,
against MLX 0.32 (`mlx_lm.benchmark`) on the same models: 512-token prompt, 128
generated tokens, both warmed up, interleaved runs after a cool-down:

| Model | Size | Generation, Skyla | Generation, MLX | Prompt, Skyla | Prompt, MLX |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B (Q8_0 / MLX 8-bit) | 0.6 GB | **160** | 167 | **4 134** | 8 065 |
| Qwen3-4B-Instruct-2507 (Q4_K_M / MLX 4-bit) | 2.3 GB | **47.4** | 51.9 | **889** | 1 363 |
| DeepSeek-R1-0528-Qwen3-8B (Q4_K_M / MLX 4-bit) | 4.7 GB | **26.6** | 28.9 | **509** | 737 |
| Qwen3-30B-A3B-Instruct-2507 REAP-15B, MoE (Q4_K_M / MLX 4-bit) | 9.1 GB | **54.3** | 61.2 | **553** | 1 119 |

The Skyla 4B run uses the abliterated (uncensored) build of the same model, with identical
shapes; both 8B runs use the same Josiefied abliterated weights. All figures are tok/s.
The same build through MoltenVK reached 24.0 / 178 tok/s on R1-8B, so the Metal backend is
1.1× faster at generation and 2.9× at prompt processing.

- **Generation** is bound by memory bandwidth. R1-8B reads its weights at ~127 GB/s
  against 134 GB/s measured by `skyla bench-bw`. The 4–11% gap to MLX is roughly the size
  difference between Q4_K_M, which keeps the output head and part of `down` in 6.5-bit
  Q6_K, and MLX's uniform 4.5-bit weights.
- **Prompt processing** multiplies on the M5's neural accelerators through the Metal 4
  `matmul2d` tensor op, which more than doubled it over `simdgroup_matrix`, and reads its
  activations as f16. It is at 51–69% of MLX for dense models and 49% for MoE, where each
  expert sees only ~16 tokens of a 128-token chunk.

These four models and Qwen3-0.6B Q4_K_M match the CPU reference with no top-1
disagreements (worst logit error 0.08–0.50 with the default `--kv v8`).

## Target machines

| | ASUS Vivobook M1405YA (primary) | MacBook with Apple M5 |
|---|---|---|
| OS | Windows 11, AMD driver 23.19 | macOS 27 |
| Backend | Vulkan 1.3 | Metal 4 (Vulkan through MoltenVK with `--features vulkan`) |
| CPU | Ryzen 7 7730U (Zen 3, 8 cores, AVX2) | Apple M5 |
| GPU | Radeon Vega 8 iGPU (`gfx90c`, 8 CU, wave64, 32 KiB LDS) | Apple M5 10-core GPU (32-lane SIMD groups, 32 KiB threadgroup memory, neural accelerator per core) |
| Memory | 40 GB DDR4-3200 shared by CPU and GPU (8 + 32 GB modules) | 16 GB unified memory, 134 GB/s measured GPU read |
| Largest buffer | ~2 GiB allocation | 4 GiB (kept for 32-bit indexing) |

The Vega has FP16 arithmetic and 8/16-bit storage, but no int8 dot product and no matrix
cores. Every design choice below follows from that and from the shared DDR4. The kernels
were designed and tuned on the Vega. On macOS the engine swaps its GPU layer for
`skyla-metal`, which runs an MSL twin of every kernel with the same bindings and graph,
32-lane reductions, an M5-tuned decode tactic and a prefill GEMM on the neural
accelerators.

## Quick start

### Requirements

Windows (Vivobook):

- Windows 11 with an AMD Vulkan 1.3 driver
- [Rust](https://rustup.rs/) (stable, MSVC toolchain) and the Visual Studio C++ Build Tools
- [Vulkan SDK](https://vulkan.lunarg.com/) (provides `glslc` and the validation layer)

```powershell
winget install Rustlang.Rustup KhronosGroup.VulkanSDK
winget install Microsoft.VisualStudio.2022.BuildTools --override "--quiet --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

macOS (Apple silicon, checked on the M5):

- macOS 26 or later (Metal 4, for the tensor ops used by prefill)
- [Rust](https://rustup.rs/) (stable)
- `glslc` from Homebrew, still needed to build the SPIR-V kernels
- Optional: MoltenVK and the Vulkan loader, for `--features vulkan`, `skyla probe` and
  `skyla bench-bw`

```sh
brew install shaderc
brew install molten-vk vulkan-loader   # optional
```

The Metal kernels are compiled from source when a model loads, so Xcode is not needed.

### Build

```powershell
git clone https://github.com/maulanasdqn/skyla-kernel.git
cd skyla-kernel
cargo build --release
```

The command is the same on both machines. `build.rs` compiles every shader in
`skyla-shaders/shaders` with `glslc`, and the subgroup kernels twice (64 and 32 lanes);
set `GLSLC` if it is not found through `VULKAN_SDK` or `PATH`. On macOS the engine uses
Metal; `cargo build --release --features skyla-engine/vulkan` builds the MoltenVK path
instead, for A/B runs.

### Get models

Any GGUF of the `qwen3` or `qwen3moe` architecture with Q8_0, Q4_K or Q6_K weights works
(abliterated variants included, e.g. Huihui-Qwen3-30B-A3B-Instruct-2507-abliterated). Keep one
folder per model under a common directory so the browser UI can list them:

```powershell
mkdir models\r1-8b, models\qwen3-4b
curl -L -o models\r1-8b\r1-8b.gguf https://huggingface.co/mradermacher/Josiefied-DeepSeek-R1-0528-Qwen3-8B-abliterated-v1-GGUF/resolve/main/Josiefied-DeepSeek-R1-0528-Qwen3-8B-abliterated-v1.Q4_K_M.gguf
curl -L -o models\qwen3-4b\qwen3-4b.gguf https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf
```

The chat template is detected from the GGUF: DeepSeek-R1 (`<｜User｜>…<｜Assistant｜>`) or
ChatML (`<|im_start|>…<|im_end|>`, Qwen3 Instruct).

### Browser chat UI

The UI is a Leptos app compiled to WebAssembly with [trunk](https://trunkrs.dev/); the
server is axum and streams tokens over a WebSocket. The engine stays native: Vulkan does
not run in a browser.

```powershell
rustup target add wasm32-unknown-unknown
cargo install trunk --locked
cd skyla-web; trunk build --release; cd ..
./target/release/skyla serve models\qwen3-4b\qwen3-4b.gguf
```

Open <http://127.0.0.1:8080>. The header's model menu lists every GGUF under `models\`
(`--models <dir>` to point elsewhere) and switches models in a few seconds, marking
architectures Skyla cannot run yet. Replies stream with the thinking section collapsible,
**Stop** interrupts generation, **New chat** clears the context, and **Settings** holds
temperature, top-p, the reply length limit and a system prompt (for example a game
character's persona). Every reply shows its token count, speed and context use.

### iPhone app

`ios/` holds a SwiftUI chat app that runs the same engine on the iPhone's GPU through
`skyla-ffi`, a static library with a small C interface. It downloads Qwen3 4B
(abliterated) or Qwen3 0.6B inside the app, or loads any `.gguf` put in its folder in the
Files app, with a 4K context. Qwen3 4B generates at about 17.6 tok/s on an iPhone 17 Pro.

```sh
brew install xcodegen
rustup target add aarch64-apple-ios
cd ios && xcodegen generate
xcodebuild -project Skyla.xcodeproj -scheme Skyla -configuration Release \
  -destination "id=<device UDID>" -allowProvisioningUpdates build
xcrun devicectl device install app --device <device UDID> build/Build/Products/Release-iphoneos/Skyla.app
```

Set `DEVELOPMENT_TEAM` in `ios/project.yml` to your team and sign in to Xcode first. The
Xcode build compiles the Rust engine for `aarch64-apple-ios` itself. With a free Apple
account the app has to be reinstalled every 7 days.

### Terminal chat

```powershell
./target/release/skyla chat models\r1-8b\r1-8b.gguf
```

Replies stream as they are generated, with the model's `<think>` section dimmed. `/reset`
clears the conversation and an empty line quits.

```
> Berapa 17 x 23? Jawab singkat.
<think>
Hmm, user is asking for the multiplication of 17 and 23 with a request for a short answer.
...
17 × 20 = 340, and 17 × 3 = 51, so 340 + 51 = 391. Yeah, that's correct.
...
</think>

391
[18 prompt tokens in 428.5ms | 253 tokens at 7.36 tok/s | context 271/16384]
```

(`--temp 0`, thinking shortened here.)

## Commands

| Command | What it does |
|---|---|
| `skyla serve <model>` | Browser chat UI on `--addr` (default `127.0.0.1:8080`); `--models <dir>` sets the folder listed in the model menu, `--web <dir>` the built UI |
| `skyla chat <model>` | Terminal chat with the model's own template (DeepSeek-R1 or ChatML) |
| `skyla run <model> --tokens <ids>` | Greedy decode from token ids; `--check` compares every step with the CPU reference, `--profile` times every GPU stage |
| `skyla ref <model> --tokens <ids>` | Greedy decode with the CPU reference; `--expect <ids>` teacher-forces another engine's output |
| `skyla tokenize <model> <text>` | Token ids from the model's tokenizer |
| `skyla gguf-info <model>` | Metadata and tensor summary |
| `skyla probe` | Vulkan device, features, limits and memory types |
| `skyla bench-bw` | Read bandwidth of every GPU memory type and of the CPU |

`chat` options:

| Option | Default | |
|---|---|---|
| `--ctx` | 16384 | Context length (KV cache size) |
| `--kv` | `v8` | KV cache precision: `f16`, `v8` (f16 keys, int8 values) or `q8` |
| `--max-tokens` | 4096 | Tokens per reply, thinking included |
| `--temp` / `--top-k` / `--top-p` | 0.6 / 40 / 0.95 | DeepSeek's recommended R1 sampling; `--temp 0` is greedy |
| `--seed` | time | Sampling seed |
| `--system` | none | System prompt |

Environment: `SKYLA_VALIDATION=1` enables the Khronos validation layer, `SKYLA_DEVICE=<n>`
selects a device other than the first integrated GPU. On macOS the Vulkan loader is found
in `/opt/homebrew/lib`, `/usr/local/lib` or `$VULKAN_SDK/lib`; Homebrew's validation layer
also needs `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib`.

## How it works

```text
GGUF checkpoint (memory-mapped)
      │
      ▼
load      weights repacked into kernel-native layouts across buffers sized to the device limit
          activations planned at fixed offsets in one arena
          RoPE/YaRN frequencies and norm weights precomputed into tables
      │
      ▼
record    one command buffer for a decode step, one for a 128-token prefill chunk;
          token, position and chunk size live in GPU memory, so recordings never change
      │
      ▼
replay    per token: dequantize the embedding row on the CPU, replay, read logits
```

### Kernels

| Kernel | Design |
|---|---|
| Matrix-vector (decode) | One subgroup (wave64 on the Vega, 32 lanes on the M5) per group of rows; each lane streams 16 bytes of weights per step and keeps a slice of the input in registers across rows. Per-format tactics: Q4_K/Q6_K 2 waves × 4 rows on the Vega and 8 SIMD groups × 1 row on the M5, Q8_0 4 × 2 |
| Matrix-matrix (prefill), Vega | 128 rows × 32 tokens per workgroup (a 128-token chunk is 4 column tiles; for MoE, one grid row per expert over the tokens routed to it); each 64-wide K step dequantizes once into LDS as f16; 4×4 micro-tiles accumulate with packed f16 FMA inside a step and f32 across steps |
| Matrix-matrix (prefill), M5 | 128 rows × 64 tokens per threadgroup (32 for MoE experts); activations are converted to f16 once per GEMM; each 64-wide K step dequantizes weights into threadgroup memory as f16 and multiplies with the Metal 4 `matmul2d` tensor op on the neural accelerators, accumulating in a cooperative tensor across K |
| Attention | Split-K decode attention (64-position chunks) with online softmax; one workgroup serves all query heads of a KV head (GQA); causal per token in prefill |
| QK-norm + RoPE | Fused per head, writes the KV cache, quantizes keys/values in registers when int8 |
| RMSNorm, SiLU·mul, merge | Small kernels; RMSNorm uses one 1024-thread workgroup |

### Weight layouts

GGUF blocks are split into aligned regions so every load is a full `uvec4`, with no
change in bytes per weight:

| Format | Block | Regions |
|---|---|---|
| Q8_0 | 34 B / 32 weights | int8 quants · f16 scales |
| Q4_K | 144 B / 256 weights | nibbles · 16-byte header (d, dmin, packed 6-bit scales) |
| Q6_K | 210 B / 256 weights | low nibbles · high bits · int8 scales · f16 d |

### KV cache

Keys and values are stored as f16 or as int8 with one f16 scale per 32 elements. Qwen3
keys carry outliers near 200 after QK-norm, so int8 keys are noticeably less accurate;
values are not. The default keeps keys in f16 and values in int8, which fits a 16k context
in 1.84 GiB.

| `--kv` | Worst logit error on R1-8B | Decode at 4k context | 16k context |
|---|---:|---:|---|
| `f16` | 0.15 | 6.57 tok/s | 2.3 GiB, over the allocation cap |
| `v8` | 0.19 | 6.72 tok/s | 1.84 GiB |
| `q8` | 2.01 | 6.88 tok/s | 1.28 GiB |

## Correctness

`skyla-ref` is a deliberately plain f32 implementation of the qwen3 forward pass. Every GPU
change is checked with `skyla run --check`, which runs both in lockstep and reports the top
token and the largest logit difference at every step, with the Vulkan validation layer on.

- The CPU reference matches llama.cpp's Vulkan backend on R1-8B to 3–4 decimals of
  probability. llama.cpp's CPU backend disagrees with both on this model.
- The GPU engine matches the reference on every greedy token in the recorded checks;
  the worst logit difference with the default settings is under 0.2. On the M5 the
  Metal backend matches the reference with no top-1 disagreements on all five test
  models, including the MoE; the MoltenVK path also runs clean under the validation
  layer.
- The tokenizer produces ids identical to `llama-tokenize` on emoji, CJK, code, CRLF,
  contractions, numbers and full chat prompts.

## Findings along the way

- **Memory placement decides bandwidth.** The 8 + 32 GB modules run dual-channel only for
  the first 16 GB, and Windows decides where pages land, so measured bandwidth moves
  between runs (35–42 GB/s). Performance decisions are made only from interleaved A/B runs
  of separately built binaries.
- **Fusion loses on Vega.** Folding RMSNorm or SiLU into the next matrix-vector kernel was
  20% and 14% slower: extra loads and registers lower occupancy in kernels that need every
  resident wave to hide DRAM latency. Moving activations to device-local memory was 2%
  slower. All three are recorded as rejected tactics.
- **YaRN is on even for short prompts.** The R1 distill uses YaRN RoPE scaling (factor 4,
  `mscale` 1.139), which changes outputs from the first token.
- **`subgroupClusteredMax(x, 32)` is broken on AMD driver 23.19** for the upper half of a
  64-lane wave; Skyla uses an explicit `subgroupShuffleXor` butterfly instead.
- **wave64 was baked into every reduction.** On the M5's 32-lane SIMD groups each
  `subgroupAdd` summed half a row and the first run disagreed on every token (logit error
  26); the subgroup kernels now take the lane count as a build-time variant.
- **MoltenVK cannot timestamp around a single dispatch.** `bench-bw` first reported
  6500 GB/s on the M5; it now times consecutive bottom-of-pipe stamps (134 GB/s).
- **`simdgroup_matrix` tops out near 3 TFLOPS on the M5.** With it, 80% of prefill GEMM
  time was the multiply itself, whatever the tile shape or alignment; the Metal 4
  `matmul2d` tensor op runs on the neural accelerators and more than doubled prefill.
  Once the multiply was fast, re-reading f32 activations for every row tile was the next
  largest cost, which 128-row tiles halved and f16 activations trimmed further.
- **A 9 GB model on a 16 GB Mac is held once, not twice.** Weights are read with
  `F_NOCACHE` straight into GPU buffers instead of being copied from the memory map, and
  loading the MoE went from 14 s to 4.2 s. When other apps already fill memory, the first
  prompt after loading it can still take seconds while macOS decompresses weight pages.
- **A hot or charging MacBook drifts by up to 30%.** M5 numbers come from interleaved
  runs after a cool-down; `skyla run --warmup` runs the prompt once first, as
  `mlx_lm.benchmark` does.

## Repository layout

| Crate | Role |
|---|---|
| `skyla-vk` | Minimal Vulkan layer: device, buffers, specialized pipelines with several descriptor sets, recorded and replayed command buffers, GPU timestamps |
| `skyla-metal` | Native Metal layer with the same surface, used on macOS, and the MSL twin of every kernel (`kernels/`, see its `CONVENTIONS.md`) |
| `skyla-shaders` | GLSL compute kernels, compiled to SPIR-V at build time, with 64- and 32-lane variants of the subgroup kernels |
| `skyla-gguf` | Memory-mapped GGUF reader and F32/F16/Q8_0/Q4_K/Q6_K dequantization |
| `skyla-ref` | f32 CPU reference forward pass and YaRN RoPE tables |
| `skyla-tok` | Byte-level BPE tokenizer read from GGUF metadata |
| `skyla-engine` | Weight planner, activation arena, KV layouts, recorded decode and prefill graphs |
| `skyla-chat` | Chat sessions: template detection (DeepSeek-R1, ChatML), sampling, UTF-8 streaming, `<think>` tracking |
| `skyla-types` | Messages shared by the server and the WebAssembly UI |
| `skyla-server` | axum server: WebSocket streaming with stop and reset, model catalog and switching, static UI; the engine runs on its own thread |
| `skyla-web` | Leptos chat UI compiled to WebAssembly |
| `skyla-cli` | The `skyla` binary |
| `skyla-ffi` | C interface to chat sessions (open, streaming chat, cancel, reset), built as a static library for the iOS app |
| `ios/` | SwiftUI iPhone app (XcodeGen project) |

Code conventions: crates live at the repository root, source files stay under 200 lines,
and the code carries no comments; design notes live in [docs/roadmap.md](docs/roadmap.md),
which also holds every measurement behind the numbers above.

## Limitations

- `qwen3` and `qwen3moe` models (head size 128) in three weight formats (Q8_0, Q4_K,
  Q6_K).
- Batch size 1 for generation; one conversation at a time, shared by every browser tab.
- Prefill slows on long contexts (about 40 tok/s at 4k tokens) because each query in a
  prefill chunk reads the KV cache separately.
- Earlier turns keep their `<think>` text in context, unlike the reference chat template,
  to avoid re-processing them.
- Tuned on two machines. It runs on Vulkan 1.3 GPUs with FP16, 8/16-bit storage and a
  subgroup size of 32 or 64, and on Apple silicon with Metal 4; only the Vega 8 and the
  M5 have been checked. Earlier Apple GPUs should run the Metal backend but are untested.
- On the M5, prompt processing is still behind MLX (51–69% dense, 49% MoE): load,
  dequantization and the multiply run one after another in each K step.

## Roadmap

See [docs/roadmap.md](docs/roadmap.md). Next: a larger prefill chunk for MoE on the M5, sharing KV reads across a prefill chunk, an OpenAI-compatible endpoint for game engines, and a
`skyla compile` step that runs the tactic sweeps on the target and stores the evidence.

## License

[MIT](LICENSE)
