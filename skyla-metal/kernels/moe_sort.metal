#include <metal_stdlib>
using namespace metal;

struct Push {
    uint ids_off;
    uint table_off;
    uint list_off;
    uint n_expert;
    uint n_used;
};

constant uint MAX_SLOTS = 1024;

kernel void moe_sort(
    device uint *a             [[buffer(0)]],
    constant Push &p           [[buffer(1)]],
    uint3 lid  [[thread_position_in_threadgroup]])
{
    threadgroup atomic_uint count[256];
    threadgroup uint start[256];
    threadgroup uint sid[MAX_SLOTS];

    uint t = lid.x;
    uint n = min(a[2] * p.n_used, MAX_SLOTS);
    atomic_store_explicit(&count[t], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = t; i < n; i += 256) {
        uint e = a[p.ids_off + i];
        sid[i] = e;
        atomic_fetch_add_explicit(&count[e], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (t == 0) {
        uint s = 0;
        for (uint e = 0; e < p.n_expert; e++) {
            uint c = atomic_load_explicit(&count[e], memory_order_relaxed);
            start[e] = s;
            a[p.table_off + 2 * e] = c;
            a[p.table_off + 2 * e + 1] = s;
            s += c;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = t; i < n; i += 256) {
        uint e = sid[i];
        uint rank = 0;
        for (uint j = 0; j < i; j++) {
            rank += uint(sid[j] == e);
        }
        a[p.list_off + start[e] + rank] = i;
    }
}
