#include <metal_stdlib>
using namespace metal;

#ifndef LANES
#define LANES 32
#endif

#ifndef SPEC_0
#define SPEC_0 0
#endif
#ifndef SPEC_1
#define SPEC_1 0
#endif
constant bool K_Q8 = (SPEC_0) != 0;
constant bool V_Q8 = (SPEC_1) != 0;

constant uint PAIRS = 64 / LANES;

struct Push {
    uint q_off;
    uint k_off;
    uint v_off;
    uint qn_off;
    uint kn_off;
    uint kc_off;
    uint vc_off;
    uint n_head;
    uint n_kv;
    float eps;
    uint rope_off;
    float mscale;
    uint q_stride;
    uint kv_stride;
    uint ks_off;
    uint vs_off;
};

static inline float block_max(float v) {
    v = abs(v);
    for (ushort d = 1; d < 32; d *= 2) {
        v = max(v, simd_shuffle_xor(v, d));
    }
    return v;
}

static inline void store(
    device half *kv,
    device char *kv8,
    constant Push &p,
    bool q8, uint cache, uint scales, uint pos, uint head, uint t, float y0, float y1)
{
    uint kv_dim = p.n_kv * 128;
    uint dst = cache + pos * kv_dim + head * 128;
    if (!q8) {
        kv[dst + t] = half(y0);
        kv[dst + t + 64] = half(y1);
        return;
    }
    float s0 = block_max(y0) / 127.0f;
    float s1 = block_max(y1) / 127.0f;
    kv8[dst + t] = char(s0 > 0.0f ? round(y0 / s0) : 0.0f);
    kv8[dst + t + 64] = char(s1 > 0.0f ? round(y1 / s1) : 0.0f);
    if (t % 32 == 0) {
        uint sdst = scales + pos * (kv_dim / 32) + head * 4 + t / 32;
        kv[sdst] = half(s0);
        kv[sdst + 2] = half(s1);
    }
}

kernel void qkv_prep(
    const device float *w      [[buffer(0)]],
    device float *a            [[buffer(1)]],
    const device uint *ctl     [[buffer(2)]],
    device half *kv            [[buffer(3)]],
    device char *kv8           [[buffer(4)]],
    constant Push &p           [[buffer(5)]],
    uint3 wg   [[threadgroup_position_in_grid]],
    uint3 lid  [[thread_position_in_threadgroup]])
{
    uint h = wg.x;
    uint tok = wg.y;
    if (tok >= ctl[2]) {
        return;
    }
    uint t = lid.x;
    uint pos = ctl[1] + tok;
    uint q_base = p.q_off + tok * p.q_stride;
    uint k_base = p.k_off + tok * p.kv_stride;
    uint v_base = p.v_off + tok * p.kv_stride;

    if (h >= p.n_head + p.n_kv) {
        uint vh = h - p.n_head - p.n_kv;
        for (uint k = 0; k < PAIRS; k++) {
            uint i = k * LANES + t;
            float v0 = a[v_base + vh * 128 + i];
            float v1 = a[v_base + vh * 128 + i + 64];
            store(kv, kv8, p, V_Q8, p.vc_off, p.vs_off, pos, vh, i, v0, v1);
        }
        return;
    }

    bool is_q = h < p.n_head;
    uint head = is_q ? h : h - p.n_head;
    uint src = (is_q ? q_base : k_base) + head * 128;
    uint nw = is_q ? p.qn_off : p.kn_off;

    float x0[PAIRS];
    float x1[PAIRS];
    float ss = 0.0f;
    for (uint k = 0; k < PAIRS; k++) {
        x0[k] = a[src + k * LANES + t];
        x1[k] = a[src + k * LANES + t + 64];
        ss += x0[k] * x0[k] + x1[k] * x1[k];
    }
    float r = rsqrt(simd_sum(ss) / 128.0f + p.eps);

    for (uint k = 0; k < PAIRS; k++) {
        uint i = k * LANES + t;
        float n0 = x0[k] * r * w[nw + i];
        float n1 = x1[k] * r * w[nw + i + 64];

        float angle = float(pos) * w[p.rope_off + i];
        float c = cos(angle) * p.mscale;
        float s = sin(angle) * p.mscale;
        float y0 = n0 * c - n1 * s;
        float y1 = n0 * s + n1 * c;

        if (is_q) {
            a[src + i] = y0;
            a[src + i + 64] = y1;
        } else {
            store(kv, kv8, p, K_Q8, p.kc_off, p.ks_off, pos, head, i, y0, y1);
        }
    }
}
