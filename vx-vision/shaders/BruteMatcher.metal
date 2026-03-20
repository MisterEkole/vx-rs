#include <metal_stdlib>
using namespace metal;

// ================================================================
// Brute-force descriptor matching (general-purpose).
//
// brutematch_hamming: Compute full Hamming distance matrix between
//   two sets of ORB descriptors (8×u32 = 256 bits each).
//   2D dispatch: one thread per (query, train) pair.
//
// brutematch_extract: For each query descriptor, find the best and
//   second-best match in the train set. Applies distance threshold
//   and Lowe's ratio test.
//   1D dispatch: one thread per query descriptor.
//
// Unlike stereo matching, this has NO epipolar/disparity constraints.
// ================================================================

struct MatcherParams {
    uint  n_query;
    uint  n_train;
    uint  max_hamming;     // maximum Hamming distance to accept
    float ratio_thresh;    // Lowe's ratio (best/second_best), e.g. 0.75
};

struct MatchResult {
    uint  query_idx;
    uint  train_idx;
    uint  distance;        // Hamming distance
    float ratio;           // best/second_best ratio
};

// Pass 1: Compute Hamming distance matrix
kernel void brutematch_hamming(
    device const uint32_t*   query_desc  [[buffer(0)]],  // n_query × 8
    device const uint32_t*   train_desc  [[buffer(1)]],  // n_train × 8
    device ushort*           dist_matrix [[buffer(2)]],   // n_query × n_train
    constant MatcherParams&  params      [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint qi = gid.x;
    uint ti = gid.y;
    if (qi >= params.n_query || ti >= params.n_train) return;

    uint dist = 0;
    for (int w = 0; w < 8; w++) {
        dist += popcount(query_desc[qi * 8 + w] ^ train_desc[ti * 8 + w]);
    }

    dist_matrix[qi * params.n_train + ti] = ushort(dist);
}

// Pass 2: Extract best match per query descriptor
kernel void brutematch_extract(
    device ushort*           dist_matrix  [[buffer(0)]],
    device MatchResult*      matches      [[buffer(1)]],
    device atomic_uint*      match_count  [[buffer(2)]],
    constant MatcherParams&  params       [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.n_query) return;

    ushort best = 0xFFFF;
    ushort second = 0xFFFF;
    uint   best_ti = 0;

    for (uint ti = 0; ti < params.n_train; ti++) {
        ushort d = dist_matrix[tid * params.n_train + ti];
        if (d < best) {
            second = best;
            best = d;
            best_ti = ti;
        } else if (d < second) {
            second = d;
        }
    }

    // Distance threshold
    if (best > ushort(params.max_hamming)) return;

    // Lowe's ratio test
    float ratio = 1.0;
    if (second > 0 && second < 0xFFFF) {
        ratio = float(best) / float(second);
        if (ratio > params.ratio_thresh) return;
    }

    uint slot = atomic_fetch_add_explicit(match_count, 1, memory_order_relaxed);
    matches[slot].query_idx = tid;
    matches[slot].train_idx = best_ti;
    matches[slot].distance  = uint(best);
    matches[slot].ratio     = ratio;
}
