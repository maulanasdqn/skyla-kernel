#include <metal_stdlib>
using namespace metal;

#ifndef LANES
#define LANES 32
#endif

struct Push {
    uint x_off;
    uint w_off;
    uint y_off;
    uint n;
    float eps;
    uint stride;
};

kernel void rmsnorm(
    const device float *w      [[buffer(0)]],
    device float *a            [[buffer(1)]],
    const device uint *ctl     [[buffer(2)]],
    constant Push &p           [[buffer(3)]],
    uint3 wg   [[threadgroup_position_in_grid]],
    uint3 lid  [[thread_position_in_threadgroup]],
    uint sg_lane [[thread_index_in_simdgroup]],
    uint sg_id   [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial[1024 / LANES];

    uint tok = wg.y;
    if (tok >= ctl[2]) {
        return;
    }
    uint x = p.x_off + tok * p.stride;
    uint y = p.y_off + tok * p.stride;
    uint t = lid.x;

    float ss = 0.0f;
    for (uint i = t; i < p.n; i += 1024) {
        float v = a[x + i];
        ss += v * v;
    }
    ss = simd_sum(ss);
    if (sg_lane == 0) {
        partial[sg_id] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total = 0.0f;
    for (uint i = 0; i < 1024 / LANES; i++) {
        total += partial[i];
    }
    float r = rsqrt(total / float(p.n) + p.eps);
    for (uint i = t; i < p.n; i += 1024) {
        a[y + i] = a[x + i] * r * w[p.w_off + i];
    }
}
