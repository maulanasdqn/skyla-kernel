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
constant uint N_REP = SPEC_0;
constant uint CHUNK = SPEC_1;
constant bool K_Q8 = (SPEC_2) != 0;
constant bool V_Q8 = (SPEC_3) != 0;

constant uint E = 128 / LANES;

#if LANES == 64
#define VEC float2
#else
#define VEC float4
#endif

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

static inline VEC load(
    const device half2 *kv,
    const device char *kv8,
    const device half *kvs,
    constant Push &p,
    bool q8, uint cache, uint scales, uint i, uint kvh, uint lane)
{
    uint at = i * p.n_kv * 64 + kvh * 64 + lane * (E / 2);
    VEC v = VEC(0.0f);
    if (!q8) {
        for (uint e = 0; e < E / 2; e++) {
            float2 pair = float2(kv[cache / 2 + at + e]);
            v[2 * e] = pair.x;
            v[2 * e + 1] = pair.y;
        }
        return v;
    }
    float s = float(kvs[scales + i * (p.n_kv * 4) + kvh * 4 + lane * E / 32]);
    for (uint e = 0; e < E; e++) {
        v[e] = float(kv8[cache + at * 2 + e]);
    }
    return v * s;
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

    VEC q[N_REP];
    float m[N_REP];
    float l[N_REP];
    VEC acc[N_REP];
    for (uint j = 0; j < N_REP; j++) {
        uint qb = q_base + (kvh * N_REP + j) * 128 + lane * E;
        for (uint e = 0; e < E; e++) {
            q[j][e] = a[qb + e] * p.scale;
        }
        m[j] = -1e30f;
        l[j] = 0.0f;
        acc[j] = VEC(0.0f);
    }

    uint end = min(split * CHUNK + CHUNK, n_pos);
    for (uint i = split * CHUNK; i < end; i++) {
        VEC k = load(kv, kv8, kvs, p, K_Q8, p.kc_off, p.ks_off, i, kvh, lane);
        VEC v = load(kv, kv8, kvs, p, V_Q8, p.vc_off, p.vs_off, i, kvh, lane);
        for (uint j = 0; j < N_REP; j++) {
            float s = simd_sum(dot(q[j], k));
            float mn = max(m[j], s);
            float c = exp(m[j] - mn);
            float e = exp(s - mn);
            l[j] = l[j] * c + e;
            acc[j] = acc[j] * c + e * v;
            m[j] = mn;
        }
    }

    for (uint j = 0; j < N_REP; j++) {
        uint base = part_base + ((kvh * N_REP + j) * p.n_split + split) * 130;
        if (lane == 0) {
            a[base] = m[j];
            a[base + 1] = l[j];
        }
        for (uint e = 0; e < E; e++) {
            a[base + 2 + lane * E + e] = acc[j][e];
        }
    }
}
