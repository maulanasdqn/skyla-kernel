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
    uint ql_off;
    uint qh_off;
    uint sc_off;
    uint d_off;
    uint x_off;
    uint y_off;
    uint n_in;
    uint n_out;
    uint ids_off;
    uint x_stride;
    uint y_stride;
};

static inline float4 q6(uint a, uint h, uint ashift, uint hshift) {
    uint4 lo = (uint4(a) >> (uint4(0, 8, 16, 24) + ashift)) & 15u;
    uint4 hi = (uint4(h) >> (uint4(0, 8, 16, 24) + hshift)) & 3u;
    return float4(lo | (hi << 4)) - 32.0f;
}

static inline float scale(uint4 s, uint k) {
    uint word = s[k / 4];
    return float(as_type<int>(word << (24 - 8 * (k & 3))) >> 24);
}

static inline float dot16(const device float4 *x4, uint4 a, uint4 h, uint ashift, uint hshift, uint xb) {
    return dot(q6(a.x, h.x, ashift, hshift), x4[xb]) + dot(q6(a.y, h.y, ashift, hshift), x4[xb + 1])
         + dot(q6(a.z, h.z, ashift, hshift), x4[xb + 2]) + dot(q6(a.w, h.w, ashift, hshift), x4[xb + 3]);
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

kernel void matvec_q6k(
    const device uint4 *w      [[buffer(0)]],
    const device half *wd      [[buffer(1)]],
    const device float4 *x4    [[buffer(2)]],
    device float *y            [[buffer(3)]],
    const device uint *ids     [[buffer(4)]],
    constant Push &p           [[buffer(5)]],
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
    uint units = p.n_in / 64;
    uint supers = p.n_in / 256;

    float acc[ROWS];
    for (uint r = 0; r < ROWS; r++) {
        acc[r] = 0.0f;
    }

    for (uint u = lane; u < units; u += LPR) {
        uint sb = u / 4;
        uint half_ = (u % 4) / 2;
        uint li = u % 2;
        uint xb = (x_off + sb * 256 + half_ * 128 + li * 16) / 4;
        uint k = half_ * 8 + li;

        for (uint r = 0; r < ROWS; r++) {
            uint row = base + min(first + r, p.n_out - 1);
            uint rs = row * supers + sb;
            uint4 a = w[p.ql_off + rs * 8 + half_ * 4 + li];
            uint4 b = w[p.ql_off + rs * 8 + half_ * 4 + 2 + li];
            uint4 h = w[p.qh_off + rs * 4 + half_ * 2 + li];
            uint4 s = w[p.sc_off + rs];
            float t = scale(s, k) * dot16(x4, a, h, 0, 0, xb) + scale(s, k + 2) * dot16(x4, b, h, 0, 2, xb + 8)
                    + scale(s, k + 4) * dot16(x4, a, h, 4, 4, xb + 16) + scale(s, k + 6) * dot16(x4, b, h, 4, 6, xb + 24);
            acc[r] += float(wd[p.d_off + rs]) * t;
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
