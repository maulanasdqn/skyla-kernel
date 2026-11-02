#include <metal_stdlib>
using namespace metal;

struct Push {
    uint g_off;
    uint u_off;
    uint n;
    uint stride;
};

kernel void silu_mul(
    device float *a            [[buffer(0)]],
    const device uint *ctl     [[buffer(1)]],
    constant Push &p           [[buffer(2)]],
    uint3 wg   [[threadgroup_position_in_grid]],
    uint3 gid  [[thread_position_in_grid]])
{
    uint i = gid.x;
    uint tok = wg.y;
    if (i >= p.n || tok >= ctl[2]) {
        return;
    }
    uint g = p.g_off + tok * p.stride + i;
    float x = a[g];
    a[g] = x / (1.0f + exp(-x)) * a[p.u_off + tok * p.stride + i];
}
