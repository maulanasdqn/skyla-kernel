#include <metal_stdlib>
using namespace metal;

struct Push {
    uint x_off;
    uint y_off;
    uint weights_off;
    uint n;
    uint stride;
    uint n_used;
};

kernel void moe_combine(
    device float *a            [[buffer(0)]],
    constant Push &p           [[buffer(1)]],
    uint3 wg   [[threadgroup_position_in_grid]],
    uint3 gid  [[thread_position_in_grid]])
{
    uint i = gid.x;
    uint tok = wg.y;
    if (i >= p.n) {
        return;
    }
    float sum = 0.0;
    for (uint s = 0; s < p.n_used; s++) {
        uint slot = tok * p.n_used + s;
        sum += a[p.weights_off + slot] * a[p.y_off + slot * p.stride + i];
    }
    a[p.x_off + tok * p.stride + i] += sum;
}
