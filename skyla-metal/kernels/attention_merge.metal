#include <metal_stdlib>
using namespace metal;

struct Push {
    uint part_off;
    uint o_off;
    uint n_split;
    uint chunk;
    uint part_stride;
    uint o_stride;
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
    uint t = lid.x;
    uint used = (ctl[1] + tok + p.chunk) / p.chunk;
    uint first = p.part_off + tok * p.part_stride + h * p.n_split * 130;

    float big = -1e30f;
    for (uint s = 0; s < used; s++) {
        big = max(big, a[first + s * 130]);
    }

    float total = 0.0f;
    float o = 0.0f;
    for (uint s = 0; s < used; s++) {
        uint base = first + s * 130;
        float w = exp(a[base] - big);
        total += w * a[base + 1];
        o += w * a[base + 2 + t];
    }
    a[p.o_off + tok * p.o_stride + h * 128 + t] = o / total;
}
