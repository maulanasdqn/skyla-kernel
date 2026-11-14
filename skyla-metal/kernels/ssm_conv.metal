#include <metal_stdlib>
using namespace metal;

#ifndef SPEC_0
#define SPEC_0 4
#endif
constant uint KERNEL = SPEC_0;
constant uint HIST = KERNEL - 1;

struct Push {
    uint x_off;
    uint x_stride;
    uint w_off;
    uint s_off;
    uint n;
};

kernel void ssm_conv(
    const device float *w      [[buffer(0)]],
    device float *a            [[buffer(1)]],
    const device uint *ctl     [[buffer(2)]],
    constant Push &p           [[buffer(3)]],
    uint3 gid  [[thread_position_in_grid]])
{
    uint c = gid.x;
    if (c >= p.n) {
        return;
    }
    float k[KERNEL];
    for (uint j = 0; j < KERNEL; j++) {
        k[j] = w[p.w_off + c * KERNEL + j];
    }
    float s[HIST];
    for (uint j = 0; j < HIST; j++) {
        s[j] = a[p.s_off + c * HIST + j];
    }
    for (uint tok = 0; tok < ctl[2]; tok++) {
        uint at = p.x_off + tok * p.x_stride + c;
        float x = a[at];
        float y = k[HIST] * x;
        for (uint j = 0; j < HIST; j++) {
            y += k[j] * s[j];
        }
        for (uint j = 0; j + 1 < HIST; j++) {
            s[j] = s[j + 1];
        }
        s[HIST - 1] = x;
        a[at] = y / (1.0f + exp(-y));
    }
    for (uint j = 0; j < HIST; j++) {
        a[p.s_off + c * HIST + j] = s[j];
    }
}
