#include <metal_stdlib>
using namespace metal;

constant uint D = 128;

struct Push {
    uint qkv_off;
    uint qkv_stride;
    uint z_off;
    uint z_stride;
    uint alpha_off;
    uint beta_off;
    uint ab_stride;
    uint o_off;
    uint o_stride;
    uint s_off;
    uint a_off;
    uint dt_off;
    uint norm_off;
    uint n_k_head;
    float eps;
};

static inline float2 group_sum(float2 v, threadgroup float2 *red, uint sg_lane, uint sg_id) {
    v = simd_sum(v);
    if (sg_lane == 0) {
        red[sg_id] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float2 total = float2(0.0f);
    for (uint i = 0; i < D / 32; i++) {
        total += red[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

// One threadgroup per value head, one thread per state column j. The state is
// stored as S[i][j] (key dim i, value dim j) so a simdgroup reads a row contiguously.
kernel void deltanet(
    const device float *w      [[buffer(0)]],
    device float *a            [[buffer(1)]],
    const device uint *ctl     [[buffer(2)]],
    constant Push &p           [[buffer(3)]],
    uint3 wg   [[threadgroup_position_in_grid]],
    uint3 lid  [[thread_position_in_threadgroup]],
    uint sg_lane [[thread_index_in_simdgroup]],
    uint sg_id   [[simdgroup_index_in_threadgroup]])
{
    threadgroup float qs[D];
    threadgroup float ks[D];
    threadgroup float2 red[D / 32];

    uint h = wg.x;
    uint j = lid.x;
    uint kh = h % p.n_k_head;
    uint key_dim = p.n_k_head * D;
    device float *S = a + p.s_off + h * D * D;
    float A = w[p.a_off + h];
    float dt = w[p.dt_off + h];
    float nw = w[p.norm_off + j];
    float scale = rsqrt(float(D));

    for (uint tok = 0; tok < ctl[2]; tok++) {
        uint base = p.qkv_off + tok * p.qkv_stride;
        float q = a[base + kh * D + j];
        float k = a[base + key_dim + kh * D + j];
        float v = a[base + 2 * key_dim + h * D + j];
        float2 ss = group_sum(float2(q * q, k * k), red, sg_lane, sg_id);
        qs[j] = q * scale / max(sqrt(ss.x), p.eps);
        ks[j] = k / max(sqrt(ss.y), p.eps);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float beta = 1.0f / (1.0f + exp(-a[p.beta_off + tok * p.ab_stride + h]));
        float x = a[p.alpha_off + tok * p.ab_stride + h] + dt;
        float decay = exp((x > 20.0f ? x : log(1.0f + exp(x))) * A);

        float kv = 0.0f;
        for (uint i = 0; i < D; i++) {
            kv += S[i * D + j] * ks[i];
        }
        float delta = (v - kv * decay) * beta;
        float o = 0.0f;
        for (uint i = 0; i < D; i++) {
            float s = S[i * D + j] * decay + ks[i] * delta;
            S[i * D + j] = s;
            o += s * qs[i];
        }

        float r = rsqrt(group_sum(float2(o * o, 0.0f), red, sg_lane, sg_id).x / float(D) + p.eps);
        float z = a[p.z_off + tok * p.z_stride + h * D + j];
        a[p.o_off + tok * p.o_stride + h * D + j] = o * r * nw * z / (1.0f + exp(-z));
    }
}
