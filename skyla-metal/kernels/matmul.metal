#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

#ifndef SPEC_0
#define SPEC_0 1
#endif
#ifndef SPEC_1
#define SPEC_1 0
#endif
#ifndef SPEC_2
#define SPEC_2 0
#endif
constant uint TYPE = SPEC_0;
constant bool ACCUM = (SPEC_1) != 0;
constant bool MOE = (SPEC_2) != 0;
#ifndef SPEC_3
#define SPEC_3 0
#endif
constant bool X_HALF = (SPEC_3) != 0;

constant uint TM = 128;
constant uint NT = 256;
#ifndef TILE_N_MOE
#define TILE_N_MOE 32
#endif
#if SPEC_2
#define TILE_N TILE_N_MOE
#endif
#ifndef TILE_N
#define TILE_N 64
#endif
constant uint TN = TILE_N;
constant uint TK = 64;
constant uint PAD = TK / 2 + 4;
constant uint XK = TK + 8;

struct Push {
    uint o0;
    uint o1;
    uint o2;
    uint o3;
    uint x_off;
    uint y_off;
    uint n_in;
    uint n_out;
    uint x_stride;
    uint y_stride;
    uint table_off;
    uint list_off;
    uint x_div;
};

static inline uint ubyte(uint4 v, uint i) {
    return (v[i / 4] >> (8 * (i % 4))) & 255u;
}

static inline float sbyte(uint4 v, uint i) {
    return float(int(ubyte(v, i) << 24) >> 24);
}

static inline float2 scale_min(uint4 h, uint j) {
    uint4 s = uint4(h.y, h.z, h.w, 0u);
    if (j < 4) {
        return float2(float(ubyte(s, j) & 63u), float(ubyte(s, j + 4) & 63u));
    }
    uint sc = (ubyte(s, j + 4) & 15u) | ((ubyte(s, j - 4) >> 6) << 4);
    uint m = (ubyte(s, j + 4) >> 4) | ((ubyte(s, j) >> 6) << 4);
    return float2(float(sc), float(m));
}

static inline void dequant(const device uint4 *w, const device half *wd, constant Push &p,
                           threadgroup half2 *ws, uint row, uint s, uint dst) {
    float v[32];
    if (TYPE == 0) {
        uint q = p.o0 + row * (p.n_in / 16) + s / 16;
        uint4 a = w[q];
        uint4 b = w[q + 1];
        float d = float(wd[p.o1 + row * (p.n_in / 32) + s / 32]);
        #pragma unroll
        for (uint i = 0; i < 16; i++) {
            v[i] = d * sbyte(a, i);
            v[16 + i] = d * sbyte(b, i);
        }
    } else if (TYPE == 1) {
        uint sb = s / 256;
        uint sub = (s % 256) / 32;
        uint q = p.o0 + row * (p.n_in / 32) + sb * 8 + (sub / 2) * 2;
        uint4 a = w[q];
        uint4 b = w[q + 1];
        uint4 h = w[p.o1 + row * (p.n_in / 256) + sb];
        float2 dm = float2(as_type<half2>(h.x));
        float2 sm = scale_min(h, sub);
        float d = dm.x * sm.x;
        float m = dm.y * sm.y;
        uint shift = (sub % 2) * 4;
        #pragma unroll
        for (uint i = 0; i < 16; i++) {
            v[i] = d * float((ubyte(a, i) >> shift) & 15u) - m;
            v[16 + i] = d * float((ubyte(b, i) >> shift) & 15u) - m;
        }
    } else {
        uint rs = row * (p.n_in / 256) + s / 256;
        uint ph = (s % 256) / 128;
        uint g = (s % 128) / 32;
        uint ql = p.o0 + rs * 8 + ph * 4 + (g & 1) * 2;
        uint4 a = w[ql];
        uint4 b = w[ql + 1];
        uint4 ha = w[p.o1 + rs * 4 + ph * 2];
        uint4 hb = w[p.o1 + rs * 4 + ph * 2 + 1];
        uint4 sc = w[p.o2 + rs];
        float d = float(wd[p.o3 + rs]);
        float s0 = d * sbyte(sc, ph * 8 + 2 * g);
        float s1 = d * sbyte(sc, ph * 8 + 2 * g + 1);
        uint ns = (g / 2) * 4;
        uint hs = 2 * g;
        #pragma unroll
        for (uint i = 0; i < 16; i++) {
            uint q0 = ((ubyte(a, i) >> ns) & 15u) | (((ubyte(ha, i) >> hs) & 3u) << 4);
            uint q1 = ((ubyte(b, i) >> ns) & 15u) | (((ubyte(hb, i) >> hs) & 3u) << 4);
            v[i] = s0 * (float(q0) - 32.0f);
            v[16 + i] = s1 * (float(q1) - 32.0f);
        }
    }
    #pragma unroll
    for (uint i = 0; i < 16; i++) {
        ws[dst + i] = half2(half(v[2 * i]), half(v[2 * i + 1]));
    }
}

kernel void matmul(
    const device uint4 *w      [[buffer(0)]],
    const device half *wd      [[buffer(1)]],
    const device float4 *x4    [[buffer(2)]],
    device float *y            [[buffer(3)]],
    const device uint *ctl     [[buffer(4)]],
    constant Push &p           [[buffer(5)]],
    uint3 wg   [[threadgroup_position_in_grid]],
    uint3 lid  [[thread_position_in_threadgroup]])
{
    constexpr uint WS_BYTES = TM * PAD * 4;
    constexpr uint XS_BYTES = TN * XK * 2;
    constexpr uint STAGE_BYTES = TM * TN * 4;
    constexpr uint SMEM_BYTES = WS_BYTES + XS_BYTES > STAGE_BYTES ? WS_BYTES + XS_BYTES : STAGE_BYTES;
    threadgroup uchar smem[SMEM_BYTES];
    threadgroup half2 *ws = (threadgroup half2 *)smem;
    threadgroup half *xs = (threadgroup half *)(smem + WS_BYTES);

    uint t = lid.x;
    uint row0 = wg.x * TM;
    uint e = wg.y;
    uint n_all = MOE ? ctl[p.table_off + 2 * e] : ctl[2];
    uint col0 = wg.z * TN;
    if (col0 >= n_all) {
        return;
    }
    uint n_tok = min(n_all - col0, TN);
    uint list = p.list_off + col0 + (MOE ? ctl[p.table_off + 2 * e + 1] : 0);
    uint base = MOE ? e * p.n_out : 0;
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(int(TM), int(TN), int(TK), false, true, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, execution_simdgroups<NT / 32>> op;
    using Tile = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>;
    Tile ta((threadgroup half *)ws, dextents<int32_t, 2>(TK, TM), array<int32_t, 2>{1, int(2 * PAD)});
    Tile tb(xs, dextents<int32_t, 2>(TK, TN), array<int32_t, 2>{1, int(XK)});
    auto acc = op.get_destination_cooperative_tensor<Tile, Tile, float>();
    #pragma unroll
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (acc.is_valid_element(i)) {
            acc[i] = 0.0f;
        }
    }

    uint lr = t / 2;
    uint hk = t % 2;
    uint grow = base + min(row0 + lr, p.n_out - 1);
    uint xt = t / 8;
    uint xk = (t % 8) * 8;
    uint xr[TN / (NT / 8)];
    #pragma unroll
    for (uint h = 0; h < TN / (NT / 8); h++) {
        xr[h] = MOE ? ctl[list + min(xt + (NT / 8) * h, n_tok - 1)] / p.x_div : col0 + xt + (NT / 8) * h;
    }

    for (uint k0 = 0; k0 < p.n_in; k0 += TK) {
        dequant(w, wd, p, ws, grow, k0 + hk * 32, lr * PAD + hk * 16);
        #pragma unroll
        for (uint h = 0; h < TN / (NT / 8); h++) {
            uint xb = (p.x_off + xr[h] * p.x_stride + k0 + xk) / 4;
            threadgroup half4 *dst = (threadgroup half4 *)(xs + (xt + (NT / 8) * h) * XK + xk);
            if (X_HALF) {
                const device half4 *xh = (const device half4 *)x4;
                dst[0] = xh[xb];
                dst[1] = xh[xb + 1];
            } else {
                dst[0] = half4(x4[xb]);
                dst[1] = half4(x4[xb + 1]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        op.run(ta, tb, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float *stage = (threadgroup float *)smem;
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tc(stage, dextents<int32_t, 2>(TN, TM), array<int32_t, 2>{1, int(TN)});
    acc.store(tc);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = t; i < TM * TN; i += NT) {
        uint tok = i / TM;
        uint lr_out = i % TM;
        uint row = row0 + lr_out;
        if (row < p.n_out && tok < n_tok) {
            uint yr = MOE ? ctl[list + tok] : col0 + tok;
            uint o = p.y_off + yr * p.y_stride + row;
            float v = stage[lr_out * TN + tok];
            y[o] = ACCUM ? y[o] + v : v;
        }
    }
}
