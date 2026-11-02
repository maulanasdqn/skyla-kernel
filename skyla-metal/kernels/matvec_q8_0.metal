#include <metal_stdlib>
using namespace metal;

#ifndef LANES
#define LANES 32
#endif

#ifndef SPEC_1
#define SPEC_1 0
#endif
#ifndef SPEC_2
#define SPEC_2 4
#endif
#ifndef SPEC_3
#define SPEC_3 0
#endif
constant bool ACCUM = (SPEC_1) != 0;
constant uint ROWS = SPEC_2;
constant bool MOE = (SPEC_3) != 0;

struct Push {
    uint q_off;
    uint d_off;
    uint x_off;
    uint y_off;
    uint n_in;
    uint n_out;
    uint ids_off;
    uint x_stride;
    uint y_stride;
};

static inline float4 i8x4(uint w) {
    return float4(as_type<int4>(uint4(w << 24, w << 16, w << 8, w)) >> 24);
}

kernel void matvec_q8_0(
    const device uint4 *wq     [[buffer(0)]],
    const device half *wd      [[buffer(1)]],
    const device float4 *x4    [[buffer(2)]],
    device float *y            [[buffer(3)]],
    const device uint *ids     [[buffer(4)]],
    constant Push &p           [[buffer(5)]],
    uint3 wg   [[threadgroup_position_in_grid]],
    uint3 lid  [[thread_position_in_threadgroup]],
    uint3 wgs  [[threads_per_threadgroup]])
{
    uint first = (wg.x * wgs.y + lid.y) * ROWS;
    if (first >= p.n_out) {
        return;
    }
    uint slot = wg.y;
    uint base = MOE ? ids[p.ids_off + slot] * p.n_out : 0;
    uint y_off = p.y_off + slot * p.y_stride;
    uint lane = lid.x;
    uint chunks = p.n_in / 16;
    uint blocks = p.n_in / 32;
    uint x_base = (p.x_off + slot * p.x_stride) / 4;

    float acc[ROWS];
    for (uint r = 0; r < ROWS; r++) {
        acc[r] = 0.0f;
    }

    for (uint c = lane; c < chunks; c += LANES) {
        uint xb = x_base + c * 4;
        float4 x0 = x4[xb];
        float4 x1 = x4[xb + 1];
        float4 x2 = x4[xb + 2];
        float4 x3 = x4[xb + 3];
        for (uint r = 0; r < ROWS; r++) {
            uint row = base + min(first + r, p.n_out - 1);
            uint4 w = wq[p.q_off + row * chunks + c];
            float s = dot(i8x4(w.x), x0) + dot(i8x4(w.y), x1) + dot(i8x4(w.z), x2) + dot(i8x4(w.w), x3);
            acc[r] += float(wd[p.d_off + row * blocks + c / 2]) * s;
        }
    }

    for (uint r = 0; r < ROWS; r++) {
        float total = simd_sum(acc[r]);
        uint row = first + r;
        if (lane == 0 && row < p.n_out) {
            if (ACCUM) {
                y[y_off + row] += total;
            } else {
                y[y_off + row] = total;
            }
        }
    }
}
