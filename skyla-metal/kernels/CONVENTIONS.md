# GLSL → MSL kernel conventions

Each GLSL compute shader `skyla-shaders/shaders/<name>.comp` gets a Metal twin
`skyla-metal/kernels/<name>.metal` with **identical semantics**. The host
(`skyla-metal`) binds and dispatches the Metal kernel exactly as the Vulkan
host dispatches the GLSL one, so translate mechanically and keep the logic,
indexing and control flow the same. Do not optimize; correctness first.

## File shape

```metal
#include <metal_stdlib>
using namespace metal;

#ifndef SPEC_0
#define SPEC_0 2
#endif
constant uint N_REP = SPEC_0;
...
struct Push { uint q_off; ...; float scale; ... };   // same fields, same order as the GLSL push block

kernel void <name>(
    device float *a            [[buffer(0)]],
    const device uint *ctl     [[buffer(1)]],
    ...
    constant Push &p           [[buffer(NB)]],        // NB = number of GLSL bindings (highest binding + 1)
    uint3 wg   [[threadgroup_position_in_grid]],
    uint3 lid  [[thread_position_in_threadgroup]],
    uint3 gid  [[thread_position_in_grid]],
    uint3 wgs  [[threads_per_threadgroup]],
    uint3 nwg  [[threadgroups_per_grid]],
    uint sg_lane [[thread_index_in_simdgroup]],
    uint sg_id   [[simdgroup_index_in_threadgroup]],
    uint n_sg    [[simdgroups_per_threadgroup]])
{ ... }
```

- The kernel function is named exactly like the file stem (`matvec_q4k`, `attention`, …).
- Only declare the attribute parameters you use.
- No comments are needed beyond what clarifies a non-obvious translation (the repo carries no comments).

## Specialization constants → macros

`layout(constant_id = i) const T NAME = D;` becomes

```metal
#ifndef SPEC_i
#define SPEC_i D
#endif
constant T NAME = SPEC_i;          // bool: constant bool NAME = (SPEC_i) != 0;
```

The host compiles each pipeline from source with `SPEC_0..SPEC_n` defined to the
specialization values (same order as the Vulkan `spec` array), so these are true
compile-time constants: arrays may be sized by them (`float acc[ROWS];`).

`LANES` is always defined by the host as `32` (Apple SIMD-group width). Treat it as a
macro exactly like the GLSL `LANES` macro (add `#ifndef LANES / #define LANES 32 / #endif`).

## Bindings

GLSL `layout(std430, binding = i) buffer B { T x[]; };` → `device T *x [[buffer(i)]]`,
`readonly` → `const device T *`. Type map:

| GLSL | MSL |
|---|---|
| `uint`, `int`, `float` | `uint`, `int`, `float` |
| `uvec4`, `vec4`, `vec2` | `uint4`, `float4`, `float2` |
| `float16_t`, `f16vec2`, `f16vec4` | `half`, `half2`, `half4` |
| `int8_t` | `char` |

Several GLSL bindings may point at the same buffer with different element types; keep them
as separate parameters with the same binding indices.

Push constants: `layout(push_constant) uniform Push {...} p;` → `struct Push {...}` and
`constant Push &p [[buffer(NB)]]`, NB = (highest GLSL binding index) + 1.

## Built-ins

| GLSL | MSL |
|---|---|
| `gl_WorkGroupID` | `threadgroup_position_in_grid` (uint3) |
| `gl_LocalInvocationID` | `thread_position_in_threadgroup` (uint3) |
| `gl_GlobalInvocationID` | `thread_position_in_grid` (uint3) |
| `gl_WorkGroupSize` | `threads_per_threadgroup` (uint3) |
| `gl_NumWorkGroups` | `threadgroups_per_grid` (uint3) |
| `gl_SubgroupInvocationID` | `thread_index_in_simdgroup` |
| `gl_SubgroupID` | `simdgroup_index_in_threadgroup` |
| `gl_NumSubgroups` | `simdgroups_per_threadgroup` |
| `subgroupAdd/Max/Min` | `simd_sum/simd_max/simd_min` |
| `subgroupShuffleXor(v, d)` | `simd_shuffle_xor(v, d)` |
| `barrier()` (+ `memoryBarrierShared()`) | `threadgroup_barrier(mem_flags::mem_threadgroup)` |
| `shared T x[N];` | `threadgroup T x[N];` declared at the top of the kernel body |
| `atomicAdd(shared_uint, v)` | `threadgroup atomic_uint` + `atomic_fetch_add_explicit(&x, v, memory_order_relaxed)` |
| `unpackHalf2x16(u)` | `float2(as_type<half2>(u))` |
| `floatBitsToUint(f)` / `uintBitsToFloat(u)` | `as_type<uint>(f)` / `as_type<float>(u)` |
| `inversesqrt` | `rsqrt` |
| `mod`-free integer ops, `min`, `max`, `exp`, `dot`, `fma`, `round`, `abs` | same names |
| `float16_t(x)`, `int8_t(x)` | `half(x)`, `char(x)` |
| `[[unroll]]` | `#pragma unroll` or nothing |

Vector component indexing with a variable (`v[i]`) works in MSL for `uint4`/`float4`.
GLSL `int(x) << 24 >> 24` sign tricks work the same on `int`.

Early `return`s must stay where they are in the GLSL (they are uniform per threadgroup
wherever a barrier follows).

## Checking

Compile-check every file with the default specialization values and with the values the
engine uses (look at `skyla-engine/src/kernels.rs` for the `spec` arrays):

```sh
xcrun -sdk macosx metal -std=metal3.1 -c skyla-metal/kernels/<name>.metal -o /dev/null
xcrun -sdk macosx metal -std=metal3.1 -DSPEC_0=8 -DSPEC_2=1 -c skyla-metal/kernels/<name>.metal -o /dev/null
```

It must compile without errors or warnings.
