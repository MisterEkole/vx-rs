#include <metal_stdlib>
using namespace metal;

// ================================================================
// Homography RANSAC support — GPU-parallel hypothesis scoring.
//
// homography_score: Given a 3x3 homography matrix and N point
//   correspondences, compute the reprojection error for each pair.
//   CPU runs RANSAC loop, GPU scores each hypothesis in parallel.
//
// Dispatch as 1D grid: one thread per point pair.
// ================================================================

struct HomographyParams {
    uint  n_points;
    float inlier_threshold;  // max reprojection error to count as inlier
    // 3x3 homography matrix (row-major)
    float h00, h01, h02;
    float h10, h11, h12;
    float h20, h21, h22;
};

// Point correspondence: (src_x, src_y) ↔ (dst_x, dst_y)
struct PointPair {
    float src_x;
    float src_y;
    float dst_x;
    float dst_y;
};

struct ScoreResult {
    float error;      // reprojection error for this pair
    uint  is_inlier;  // 1 if error < threshold, 0 otherwise
};

kernel void homography_score(
    device const PointPair*    pairs   [[buffer(0)]],
    device ScoreResult*        results [[buffer(1)]],
    device atomic_uint*        count   [[buffer(2)]],
    constant HomographyParams& params  [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params.n_points) return;

    PointPair p = pairs[gid];

    // Project src point through homography
    float w = params.h20 * p.src_x + params.h21 * p.src_y + params.h22;
    if (abs(w) < 1e-8) {
        results[gid].error = 1e6;
        results[gid].is_inlier = 0;
        return;
    }

    float px = (params.h00 * p.src_x + params.h01 * p.src_y + params.h02) / w;
    float py = (params.h10 * p.src_x + params.h11 * p.src_y + params.h12) / w;

    // Reprojection error
    float dx = px - p.dst_x;
    float dy = py - p.dst_y;
    float error = sqrt(dx * dx + dy * dy);

    results[gid].error = error;

    uint is_inlier = (error < params.inlier_threshold) ? 1u : 0u;
    results[gid].is_inlier = is_inlier;

    if (is_inlier) {
        atomic_fetch_add_explicit(count, 1, memory_order_relaxed);
    }
}
