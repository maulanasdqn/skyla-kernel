#include <metal_stdlib>
using namespace metal;

struct Push {
    uint src_off;
    uint dst_off;
    uint n;
    uint src_stride;
    uint dst_stride;
    uint per_token;
};

kernel void to_half(
    const device float *a      [[buffer(0)]],
    device half *h             [[buffer(1)]],
    const device uint *ctl     [[buffer(2)]],
    constant Push &p           [[buffer(3)]],
    uint3 gid  [[thread_position_in_grid]],
    uint3 wg   [[threadgroup_position_in_grid]])
{
    uint i = gid.x * 4;
    uint row = wg.y;
    if (i >= p.n || row >= ctl[2] * p.per_token) {
        return;
    }
    const device float4 *src = (const device float4 *)(a + p.src_off + row * p.src_stride + i);
    *(device half4 *)(h + p.dst_off + row * p.dst_stride + i) = half4(*src);
}
