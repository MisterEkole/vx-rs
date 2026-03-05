#include <metal_stdlib>
using namespace metal;

// Shared struct — identical layout to FastDetect.metal and HarrisResponse.metal
struct CornerPoint {
    float2 position;
    float  response;
    uint   pyramid_level;
};

struct NMSParams {
    uint  n_corners;
    float min_distance;   // suppress if a stronger neighbour is within this radius
};

// ================================================================
// Non-Maximum Suppression
//
// 1D dispatch: one thread per input corner.
// A corner survives if no other corner with a strictly higher response
// exists within min_distance pixels.  Survivors are atomically appended
// to the output buffer in arbitrary order.
// ================================================================
kernel void nms_suppress(
    device const CornerPoint* input     [[buffer(0)]],
    device CornerPoint*       output    [[buffer(1)]],
    device atomic_uint*       out_count [[buffer(2)]],
    constant NMSParams&       params    [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.n_corners) return;

    CornerPoint c = input[tid];
    float min_dist_sq = params.min_distance * params.min_distance;

    for (uint j = 0; j < params.n_corners; j++) {
        if (j == tid) continue;

        CornerPoint other = input[j];

        // Only suppress if the neighbour is strictly stronger
        if (other.response <= c.response) continue;

        float2 d = c.position - other.position;
        if (dot(d, d) < min_dist_sq) return;   // dominated — suppress
    }

    // Survived: atomically claim an output slot
    uint slot = atomic_fetch_add_explicit(out_count, 1, memory_order_relaxed);
    output[slot] = c;
}
