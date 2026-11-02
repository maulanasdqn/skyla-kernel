#include <metal_stdlib>
using namespace metal;

#ifndef LANES
#define LANES 32
#endif

struct Push {
    uint logits_off;
    uint ids_off;
    uint weights_off;
    uint n_expert;
    uint n_used;
};

kernel void moe_route(
    device float *a            [[buffer(0)]],
    device uint *ids           [[buffer(1)]],
    constant Push &p           [[buffer(2)]],
    uint3 wg   [[threadgroup_position_in_grid]],
    uint3 lid  [[thread_position_in_threadgroup]],
    uint sg_lane [[thread_index_in_simdgroup]],
    uint sg_id   [[simdgroup_index_in_threadgroup]],
    uint n_sg    [[simdgroups_per_threadgroup]])
{
    threadgroup float wave_value[256 / LANES];
    threadgroup uint wave_index[256 / LANES];
    threadgroup float chosen_value[16];
    threadgroup uint chosen_index[16];

    uint tok = wg.y;
    if (tok >= as_type<uint>(a[2])) {
        return;
    }
    uint t = lid.x;
    uint waves = n_sg;
    uint logits_off = p.logits_off + tok * p.n_expert;
    uint ids_off = p.ids_off + tok * p.n_used;
    uint weights_off = p.weights_off + tok * p.n_used;
    float v = t < p.n_expert ? a[logits_off + t] : -3e38;

    for (uint k = 0; k < p.n_used; k++) {
        float m = simd_max(v);
        uint idx = simd_min(v == m ? t : 0xffffffffu);
        if (sg_lane == 0) {
            wave_value[sg_id] = m;
            wave_index[sg_id] = idx;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t == 0) {
            uint best = 0;
            for (uint w = 1; w < waves; w++) {
                bool larger = wave_value[w] > wave_value[best];
                bool tie = wave_value[w] == wave_value[best] && wave_index[w] < wave_index[best];
                if (larger || tie) {
                    best = w;
                }
            }
            chosen_value[k] = wave_value[best];
            chosen_index[k] = wave_index[best];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t == chosen_index[k]) {
            v = -3e38;
        }
    }

    if (t != 0) {
        return;
    }
    float largest = chosen_value[0];
    float total = 0.0;
    for (uint k = 0; k < p.n_used; k++) {
        total += exp(chosen_value[k] - largest);
    }
    for (uint k = 0; k < p.n_used; k++) {
        ids[ids_off + k] = chosen_index[k];
        a[weights_off + k] = exp(chosen_value[k] - largest) / total;
    }
}
