#include <metal_stdlib>
using namespace metal;

#ifndef SPEC_0
#define SPEC_0 128
#endif
#ifndef SPEC_1
#define SPEC_1 0
#endif
constant uint HEAD_DIM = SPEC_0;
constant bool GATED = (SPEC_1) != 0;

constant uint PART = HEAD_DIM + 2;

struct Push {
    uint part_off;
    uint o_off;
    uint n_split;
    uint chunk;
    uint part_stride;
    uint o_stride;
    uint g_off;
    uint g_stride;
};

kernel void attention_merge(
    device float *a            [[buffer(0)]],
    const device uint *ctl     [[buffer(1)]],
    constant Push &p           [[buffer(2)]],
    uint3 wg   [[threadgroup_position_in_grid]],
    uint3 lid  [[thread_position_in_threadgroup]])
{
    uint h = wg.x;
    uint tok = wg.y;
    if (tok >= ctl[2]) {
        return;
    }
    uint used = (ctl[1] + tok + p.chunk) / p.chunk;
    uint first = p.part_off + tok * p.part_stride + h * p.n_split * PART;

    float big = -1e30f;
    for (uint s = 0; s < used; s++) {
        big = max(big, a[first + s * PART]);
    }

    for (uint t = lid.x; t < HEAD_DIM; t += 128) {
        float total = 0.0f;
        float o = 0.0f;
        for (uint s = 0; s < used; s++) {
            uint base = first + s * PART;
            float w = exp(a[base] - big);
            total += w * a[base + 1];
            o += w * a[base + 2 + t];
        }
        o /= total;
        if (GATED) {
            float g = a[p.g_off + tok * p.g_stride + h * 2 * HEAD_DIM + HEAD_DIM + t];
            o /= 1.0f + exp(-g);
        }
        a[p.o_off + tok * p.o_stride + h * HEAD_DIM + t] = o;
    }
}
