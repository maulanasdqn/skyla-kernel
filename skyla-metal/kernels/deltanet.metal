#include <metal_stdlib>
using namespace metal;

constant uint D = 128;
constant uint COLS = 4;
constant uint ROWS = D / 32;

struct Push {
    uint qkv_off;
    uint qkv_stride;
    uint alpha_off;
    uint beta_off;
    uint ab_stride;
    uint o_off;
    uint o_stride;
    uint s_off;
    uint a_off;
    uint dt_off;
    uint n_k_head;
    float eps;
};

static inline float4 load4(const device float *p) {
    return float4(*(const device packed_float4 *)p);
}

kernel void deltanet(
    const device float *w      [[buffer(0)]],
    device float *a            [[buffer(1)]],
    const device uint *ctl     [[buffer(2)]],
    constant Push &p           [[buffer(3)]],
    uint3 wg   [[threadgroup_position_in_grid]],
    uint sg_lane [[thread_index_in_simdgroup]],
    uint sg_id   [[simdgroup_index_in_threadgroup]],
    uint n_sg    [[simdgroups_per_threadgroup]])
{
    uint h = wg.x;
    uint col = (wg.y * n_sg + sg_id) * COLS;
    uint row = sg_lane * ROWS;
    uint kh = h % p.n_k_head;
    uint key_dim = p.n_k_head * D;
    device float *S = a + p.s_off + h * D * D + row * D + col;
    float A = w[p.a_off + h];
    float dt = w[p.dt_off + h];
    float scale = rsqrt(float(D));

    float4 s[ROWS];
    for (uint r = 0; r < ROWS; r++) {
        s[r] = load4(S + r * D);
    }

    for (uint tok = 0; tok < ctl[2]; tok++) {
        uint base = p.qkv_off + tok * p.qkv_stride;
        float4 q = load4(a + base + kh * D + row);
        float4 k = load4(a + base + key_dim + kh * D + row);
        float4 v = load4(a + base + 2 * key_dim + h * D + col);
        float2 ss = simd_sum(float2(dot(q, q), dot(k, k)));
        q *= scale / max(sqrt(ss.x), p.eps);
        k /= max(sqrt(ss.y), p.eps);

        float beta = 1.0f / (1.0f + exp(-a[p.beta_off + tok * p.ab_stride + h]));
        float x = a[p.alpha_off + tok * p.ab_stride + h] + dt;
        float decay = exp((x > 20.0f ? x : log(1.0f + exp(x))) * A);

        float4 kv = 0.0f;
        for (uint r = 0; r < ROWS; r++) {
            kv += s[r] * k[r];
        }
        float4 delta = (v - simd_sum(kv) * decay) * beta;
        float4 o = 0.0f;
        for (uint r = 0; r < ROWS; r++) {
            s[r] = s[r] * decay + k[r] * delta;
            o += s[r] * q[r];
        }
        o = simd_sum(o);
        if (sg_lane == 0) {
            *(device packed_float4 *)(a + p.o_off + tok * p.o_stride + h * D + col) = packed_float4(o);
        }
    }

    for (uint r = 0; r < ROWS; r++) {
        *(device packed_float4 *)(S + r * D) = packed_float4(s[r]);
    }
}
