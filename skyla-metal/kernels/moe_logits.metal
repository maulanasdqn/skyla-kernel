#include <metal_stdlib>
using namespace metal;

#ifndef LANES
#define LANES 32
#endif

struct Push {
    uint x_off;
    uint router_off;
    uint logits_off;
    uint n_embd;
    uint x_stride;
};

kernel void moe_logits(
    const device float4 *w4    [[buffer(0)]],
    device float4 *a4          [[buffer(1)]],
    device float *logits       [[buffer(2)]],
    constant Push &p           [[buffer(3)]],
    uint3 wg   [[threadgroup_position_in_grid]],
    uint3 lid  [[thread_position_in_threadgroup]],
    uint3 nwg  [[threadgroups_per_grid]])
{
    uint e = wg.x;
    uint tok = wg.y;
    uint lane = lid.x;
    uint row = (p.router_off + e * p.n_embd) / 4;
    uint x = (p.x_off + tok * p.x_stride) / 4;
    float s = 0.0f;
    for (uint i = lane; i < p.n_embd / 4; i += LANES) {
        s += dot(w4[row + i], a4[x + i]);
    }
    s = simd_sum(s);
    if (lane == 0) {
        logits[p.logits_off + tok * nwg.x + e] = s;
    }
}
