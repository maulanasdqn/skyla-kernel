#include <metal_stdlib>
using namespace metal;

struct Push { uint n_vec4; };

kernel void bandwidth(
    const device uint4 *src    [[buffer(0)]],
    device uint *dst           [[buffer(1)]],
    constant Push &p           [[buffer(2)]],
    uint3 gid  [[thread_position_in_grid]],
    uint3 wgs  [[threads_per_threadgroup]],
    uint3 nwg  [[threadgroups_per_grid]])
{
    uint g = gid.x;
    uint stride = nwg.x * wgs.x;

    uint4 acc = uint4(0);
    for (uint i = g; i < p.n_vec4; i += stride) {
        acc ^= src[i];
    }

    uint r = acc.x ^ acc.y ^ acc.z ^ acc.w;
    if (r == 0x9e3779b9u) {
        dst[g] = r;
    }
}
