#include <metal_stdlib>
using namespace metal;

constant uint D = 128;

struct Push {
    uint o_off;
    uint o_stride;
    uint z_off;
    uint z_stride;
    uint norm_off;
    float eps;
};

kernel void ssm_norm(
    const device float *w      [[buffer(0)]],
    device float *a            [[buffer(1)]],
    const device uint *ctl     [[buffer(2)]],
    constant Push &p           [[buffer(3)]],
    uint3 wg   [[threadgroup_position_in_grid]],
    uint3 lid  [[thread_position_in_threadgroup]],
    uint sg_lane [[thread_index_in_simdgroup]],
    uint sg_id   [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[D / 32];

    uint h = wg.x;
    uint tok = wg.y;
    if (tok >= ctl[2]) {
        return;
    }
    uint j = lid.x;
    uint at = p.o_off + tok * p.o_stride + h * D + j;
    float o = a[at];
    float ss = simd_sum(o * o);
    if (sg_lane == 0) {
        red[sg_id] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < D / 32; i++) {
        total += red[i];
    }
    float z = a[p.z_off + tok * p.z_stride + h * D + j];
    a[at] = o * rsqrt(total / float(D) + p.eps) * w[p.norm_off + j] * z / (1.0f + exp(-z));
}
