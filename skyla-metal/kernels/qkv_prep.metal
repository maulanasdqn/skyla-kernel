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
#ifndef SPEC_2
#define SPEC_2 128
#endif
#ifndef SPEC_3
#define SPEC_3 128
#endif
#ifndef SPEC_4
#define SPEC_4 0
#endif
constant bool K_Q8 = (SPEC_0) != 0;
constant bool V_Q8 = (SPEC_1) != 0;
constant uint HEAD_DIM = SPEC_2;
constant uint N_ROT = SPEC_3;
constant bool GATED = (SPEC_4) != 0;

constant uint E = HEAD_DIM / LANES;
constant uint ROT_K = N_ROT / 2 / LANES;
constant uint Q_HEAD = GATED ? 2 * HEAD_DIM : HEAD_DIM;

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
    bool q8, uint cache, uint scales, uint pos, uint head, uint i, float y)
{
    uint kv_dim = p.n_kv * HEAD_DIM;
    uint dst = cache + pos * kv_dim + head * HEAD_DIM + i;
    if (!q8) {
        kv[dst] = half(y);
        return;
    }
    float s = block_max(y) / 127.0f;
    kv8[dst] = char(s > 0.0f ? round(y / s) : 0.0f);
    if (i % 32 == 0) {
        kv[scales + pos * (kv_dim / 32) + head * (HEAD_DIM / 32) + i / 32] = half(s);
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
        for (uint k = 0; k < E; k++) {
            uint i = k * LANES + t;
            store(kv, kv8, p, V_Q8, p.vc_off, p.vs_off, pos, vh, i, a[v_base + vh * HEAD_DIM + i]);
        }
        return;
    }

    bool is_q = h < p.n_head;
    uint head = is_q ? h : h - p.n_head;
    uint src = is_q ? q_base + head * Q_HEAD : k_base + head * HEAD_DIM;
    uint nw = is_q ? p.qn_off : p.kn_off;

    float x[E];
    float ss = 0.0f;
    for (uint k = 0; k < E; k++) {
        x[k] = a[src + k * LANES + t];
        ss += x[k] * x[k];
    }
    float r = rsqrt(simd_sum(ss) / float(HEAD_DIM) + p.eps);
    for (uint k = 0; k < E; k++) {
        x[k] *= r * w[nw + k * LANES + t];
    }

    for (uint k = 0; k < ROT_K; k++) {
        uint i = k * LANES + t;
        float angle = float(pos) * w[p.rope_off + i];
        float c = cos(angle) * p.mscale;
        float s = sin(angle) * p.mscale;
        float x0 = x[k];
        float x1 = x[k + ROT_K];
        x[k] = x0 * c - x1 * s;
        x[k + ROT_K] = x0 * s + x1 * c;
    }

    for (uint k = 0; k < E; k++) {
        uint i = k * LANES + t;
        if (is_q) {
            a[src + i] = x[k];
        } else {
            store(kv, kv8, p, K_Q8, p.kc_off, p.ks_off, pos, head, i, x[k]);
        }
    }
}
