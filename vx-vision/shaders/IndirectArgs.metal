#include <metal_stdlib>
using namespace metal;

/// Mirrors MTLDispatchThreadgroupsIndirectArguments: three uint32 values.
struct IndirectArgs {
    uint threadgroups_x;
    uint threadgroups_y;
    uint threadgroups_z;
};

struct IndirectSetupParams {
    uint threads_per_threadgroup; // threadgroup width for the target kernel
};

/// Reads an atomic corner count and writes indirect dispatch arguments.
///
/// The target kernel (e.g. Harris) dispatches one thread per corner in a 1D grid.
/// We compute: threadgroups_x = ceil(count / threads_per_threadgroup), y=z=1.
kernel void prepare_indirect_args(
    device const atomic_uint*   count_buf  [[buffer(0)]],
    device IndirectArgs*        args       [[buffer(1)]],
    constant IndirectSetupParams& params   [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0) return;

    uint n = atomic_load_explicit(count_buf, memory_order_relaxed);
    uint tpg = params.threads_per_threadgroup;

    args->threadgroups_x = (n + tpg - 1) / tpg;
    args->threadgroups_y = 1;
    args->threadgroups_z = 1;
}
