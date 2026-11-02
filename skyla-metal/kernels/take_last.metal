#include <metal_stdlib>
using namespace metal;

struct Push {
    uint src_off;
    uint dst_off;
    uint n;
    uint stride;
};

kernel void take_last(
    device float *a            [[buffer(0)]],
    const device uint *ctl     [[buffer(1)]],
    constant Push &p           [[buffer(2)]],
    uint3 gid  [[thread_position_in_grid]])
{
    uint i = gid.x;
    if (i < p.n) {
        a[p.dst_off + i] = a[p.src_off + (ctl[2] - 1) * p.stride + i];
    }
}
