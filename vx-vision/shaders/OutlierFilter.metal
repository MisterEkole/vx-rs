#include <metal_stdlib>
using namespace metal;

// ================================================================
// Statistical outlier removal for point clouds.
// Pass 1: compute mean k-NN distance. Pass 2: classify inliers.
// Dispatch: 1D per-point.
// ================================================================

struct OutlierParams {
    uint  n_points;
    uint  k_neighbors;
    float std_ratio;
};

struct PointXYZ {
    float x, y, z;
    float _pad;
};

// Pass 1: mean distance to k-NN per point
kernel void outlier_compute_distances(
    device PointXYZ*         points     [[buffer(0)]],
    device float*            mean_dists [[buffer(1)]],
    constant OutlierParams&  params     [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.n_points) return;

    float3 center = float3(points[tid].x, points[tid].y, points[tid].z);

    // Maintain k smallest distances using a simple insertion sort
    // For small k (< 32), this is efficient enough on GPU
    float best[32];
    uint k = min(params.k_neighbors, 32u);
    for (uint i = 0; i < k; i++) {
        best[i] = 1e20;
    }

    for (uint i = 0; i < params.n_points; i++) {
        if (i == tid) continue;
        float3 p = float3(points[i].x, points[i].y, points[i].z);
        float d = distance(center, p);

        // Insert into sorted best-k array
        if (d < best[k - 1]) {
            best[k - 1] = d;
            // Bubble up
            for (int j = int(k) - 2; j >= 0; j--) {
                if (best[j + 1] < best[j]) {
                    float tmp = best[j];
                    best[j] = best[j + 1];
                    best[j + 1] = tmp;
                } else {
                    break;
                }
            }
        }
    }

    // Mean distance to k nearest neighbors
    float sum = 0.0;
    uint valid = 0;
    for (uint i = 0; i < k; i++) {
        if (best[i] < 1e19) {
            sum += best[i];
            valid++;
        }
    }

    mean_dists[tid] = (valid > 0) ? (sum / float(valid)) : 0.0;
}

// Pass 2: classify inliers vs outliers
kernel void outlier_classify(
    device float*           mean_dists  [[buffer(0)]],
    device uint*            inlier_mask [[buffer(1)]],  // 1 = inlier, 0 = outlier
    device atomic_uint*     inlier_count [[buffer(2)]],
    constant OutlierParams& params      [[buffer(3)]],
    constant float&         threshold   [[buffer(4)]],  // precomputed: mean + std_ratio * stddev
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.n_points) return;

    if (mean_dists[tid] <= threshold) {
        inlier_mask[tid] = 1;
        atomic_fetch_add_explicit(inlier_count, 1, memory_order_relaxed);
    } else {
        inlier_mask[tid] = 0;
    }
}
