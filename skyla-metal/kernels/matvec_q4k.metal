#include <metal_stdlib>
using namespace metal;

#ifndef LANES
#define LANES 32
#endif

#ifndef SPEC_1
#define SPEC_1 0
#endif
#ifndef SPEC_2
#define SPEC_2 2
#endif
#ifndef SPEC_3
#define SPEC_3 0
#endif
#ifndef SPEC_4
#define SPEC_4 LANES
#endif
constant bool ACCUM = (SPEC_1) != 0;
constant uint ROWS = SPEC_2;
constant bool MOE = (SPEC_3) != 0;
constant uint LPR = SPEC_4;

struct Push {
    uint qs_off;
    uint hd_off;
    uint x_off;
    uint y_off;
    uint n_in;
    uint n_out;
    uint ids_off;
    uint x_stride;
    uint y_stride;
};

static inline float4 lo4(uint v) {
    return float4(uint4(v & 15u, (v >> 8) & 15u, (v >> 16) & 15u, (v >> 24) & 15u));
}

static inline float4 hi4(uint v) {
    return float4(uint4((v >> 4) & 15u, (v >> 12) & 15u, (v >> 20) & 15u, v >> 28));
}

static inline uint sbyte(uint4 h, uint k) {
    uint word = k < 4 ? h.y : (k < 8 ? h.z : h.w);
    return (word >> (8 * (k & 3))) & 255u;
}

static inline float2 scale_min(uint4 h, uint j) {
    if (j < 4) {
        return float2(uint2(sbyte(h, j) & 63u, sbyte(h, j + 4) & 63u));
    }
    uint s = (sbyte(h, j + 4) & 15u) | ((sbyte(h, j - 4) >> 6) << 4);
    uint m = (sbyte(h, j + 4) >> 4) | ((sbyte(h, j) >> 6) << 4);
    return float2(uint2(s, m));
}

static inline float reduce(float v) {
    if (LPR == LANES) {
        return simd_sum(v);
    }
    for (uint d = LPR / 2; d > 0; d /= 2) {
        v += simd_shuffle_xor(v, ushort(d));
    }
    return v;
}

kernel void matvec_q4k(
    const device uint4 *w      [[buffer(0)]],
    const device float4 *x4    [[buffer(1)]],
    device float *y            [[buffer(2)]],
    const device uint *ids     [[buffer(3)]],
    constant Push &p           [[buffer(4)]],
    uint3 wg   [[threadgroup_position_in_grid]],
    uint3 lid  [[thread_position_in_threadgroup]],
    uint3 wgs  [[threads_per_threadgroup]])
{
    uint lane = lid.x % LPR;
    uint sub = lid.x / LPR;
    uint wave = wg.x * wgs.y + lid.y;
    uint first = (wave * (LANES / LPR) + sub) * ROWS;
    if (LPR == LANES && first >= p.n_out) {
        return;
    }
    uint slot = wg.y;
    uint base = MOE ? ids[p.ids_off + slot] * p.n_out : 0;
    uint x_off = p.x_off + slot * p.x_stride;
    uint y_off = p.y_off + slot * p.y_stride;
    uint units = p.n_in / 32;
    uint supers = p.n_in / 256;

    float acc[ROWS];
    for (uint r = 0; r < ROWS; r++) {
        acc[r] = 0.0f;
    }

    for (uint u = lane; u < units; u += LPR) {
        uint sb = u / 8;
        uint j = (u % 8) / 2;
        uint xb = (x_off + sb * 256 + j * 64 + (u % 2) * 16) / 4;
        float4 l0 = x4[xb], l1 = x4[xb + 1], l2 = x4[xb + 2], l3 = x4[xb + 3];
        float4 h0 = x4[xb + 8], h1 = x4[xb + 9], h2 = x4[xb + 10], h3 = x4[xb + 11];
        float4 one = float4(1.0f);
        float sum_lo = dot(l0 + l1 + l2 + l3, one);
        float sum_hi = dot(h0 + h1 + h2 + h3, one);

        for (uint r = 0; r < ROWS; r++) {
            uint row = base + min(first + r, p.n_out - 1);
            uint4 q = w[p.qs_off + row * units + u];
            uint4 h = w[p.hd_off + row * supers + sb];
            float2 dm = float2(as_type<half2>(h.x));
            float2 a = scale_min(h, 2 * j);
            float2 b = scale_min(h, 2 * j + 1);
            float dl = dot(lo4(q.x), l0) + dot(lo4(q.y), l1) + dot(lo4(q.z), l2) + dot(lo4(q.w), l3);
            float dh = dot(hi4(q.x), h0) + dot(hi4(q.y), h1) + dot(hi4(q.z), h2) + dot(hi4(q.w), h3);
            acc[r] += dm.x * (a.x * dl + b.x * dh) - dm.y * (a.y * sum_lo + b.y * sum_hi);
        }
    }

    for (uint r = 0; r < ROWS; r++) {
        float total = reduce(acc[r]);
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
