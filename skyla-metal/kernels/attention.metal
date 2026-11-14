#include <metal_stdlib>
using namespace metal;

#ifndef LANES
#define LANES 32
#endif

#ifndef SPEC_0
#define SPEC_0 2
#endif
#ifndef SPEC_1
#define SPEC_1 64
#endif
#ifndef SPEC_2
#define SPEC_2 0
#endif
#ifndef SPEC_3
#define SPEC_3 0
#endif
#ifndef SPEC_4
#define SPEC_4 128
#endif
#ifndef SPEC_5
#define SPEC_5 128
#endif
constant uint N_REP = SPEC_0;
constant uint CHUNK = SPEC_1;
constant bool K_Q8 = (SPEC_2) != 0;
constant bool V_Q8 = (SPEC_3) != 0;
constant uint HEAD_DIM = SPEC_4;
constant uint Q_HEAD = SPEC_5;

constant uint E = HEAD_DIM / LANES;
constant uint NV = E / 4;
constant uint PART = HEAD_DIM + 2;

struct Push {
    uint q_off;
    uint part_off;
    uint kc_off;
    uint vc_off;
    uint n_kv;
    uint n_split;
    float scale;
    uint q_stride;
    uint part_stride;
    uint ks_off;
    uint vs_off;
};

static inline void load(
    thread float4 *v,
    const device half2 *kv,
    const device char *kv8,
    const device half *kvs,
    constant Push &p,
    bool q8, uint cache, uint scales, uint i, uint kvh, uint lane)
{
    uint at = i * p.n_kv * (HEAD_DIM / 2) + kvh * (HEAD_DIM / 2) + lane * (E / 2);
    if (!q8) {
        for (uint n = 0; n < NV; n++) {
            float2 lo = float2(kv[cache / 2 + at + 2 * n]);
            float2 hi = float2(kv[cache / 2 + at + 2 * n + 1]);
            v[n] = float4(lo, hi);
        }
        return;
    }
    float s = float(kvs[scales + i * (p.n_kv * HEAD_DIM / 32) + kvh * (HEAD_DIM / 32) + lane * E / 32]);
    for (uint n = 0; n < NV; n++) {
        uint b = cache + at * 2 + 4 * n;
        v[n] = float4(kv8[b], kv8[b + 1], kv8[b + 2], kv8[b + 3]) * s;
    }
}

kernel void attention(
    device float *a            [[buffer(0)]],
    const device uint *ctl     [[buffer(1)]],
    const device half2 *kv     [[buffer(2)]],
    const device char *kv8     [[buffer(3)]],
    const device half *kvs     [[buffer(4)]],
    constant Push &p           [[buffer(5)]],
    uint3 wg   [[threadgroup_position_in_grid]],
    uint3 lid  [[thread_position_in_threadgroup]])
{
    uint kvh = wg.x;
    uint split = wg.y;
    uint tok = wg.z;
    uint n_pos = ctl[1] + tok + 1;
    if (tok >= ctl[2] || split * CHUNK >= n_pos) {
        return;
    }
    uint lane = lid.x;
    uint q_base = p.q_off + tok * p.q_stride;
    uint part_base = p.part_off + tok * p.part_stride;

    float4 q[N_REP][NV];
    float m[N_REP];
    float l[N_REP];
    float4 acc[N_REP][NV];
    for (uint j = 0; j < N_REP; j++) {
        uint qb = q_base + (kvh * N_REP + j) * Q_HEAD + lane * E;
        for (uint n = 0; n < NV; n++) {
            q[j][n] = float4(a[qb + 4 * n], a[qb + 4 * n + 1], a[qb + 4 * n + 2], a[qb + 4 * n + 3]) * p.scale;
            acc[j][n] = float4(0.0f);
        }
        m[j] = -1e30f;
        l[j] = 0.0f;
    }

    uint end = min(split * CHUNK + CHUNK, n_pos);
    for (uint i = split * CHUNK; i < end; i++) {
        float4 k[NV];
        float4 v[NV];
        load(k, kv, kv8, kvs, p, K_Q8, p.kc_off, p.ks_off, i, kvh, lane);
        load(v, kv, kv8, kvs, p, V_Q8, p.vc_off, p.vs_off, i, kvh, lane);
        for (uint j = 0; j < N_REP; j++) {
            float d = 0.0f;
            for (uint n = 0; n < NV; n++) {
                d += dot(q[j][n], k[n]);
            }
            float s = simd_sum(d);
            float mn = max(m[j], s);
            float c = exp(m[j] - mn);
            float e = exp(s - mn);
            l[j] = l[j] * c + e;
            for (uint n = 0; n < NV; n++) {
                acc[j][n] = acc[j][n] * c + e * v[n];
            }
            m[j] = mn;
        }
    }

    for (uint j = 0; j < N_REP; j++) {
        uint base = part_base + ((kvh * N_REP + j) * p.n_split + split) * PART;
        if (lane == 0) {
            a[base] = m[j];
            a[base + 1] = l[j];
        }
        for (uint n = 0; n < NV; n++) {
            for (uint e = 0; e < 4; e++) {
                a[base + 2 + lane * E + 4 * n + e] = acc[j][n][e];
            }
        }
    }
}
