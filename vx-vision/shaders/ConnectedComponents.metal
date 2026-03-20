#include <metal_stdlib>
using namespace metal;

// ================================================================
// Connected components labeling — iterative label propagation.
//
// ccl_init:    Initialize labels: foreground pixels get unique
//              label = y * width + x + 1. Background pixels get 0.
//
// ccl_iterate: One iteration of label propagation. Each foreground
//              pixel adopts the minimum label from its 4-connected
//              neighbors. Repeat until convergence (no changes).
//              Sets a flag if any label changed (for CPU convergence check).
//
// Dispatch as 2D grid: one thread per pixel.
// ================================================================

struct CCLParams {
    uint  width;
    uint  height;
    float threshold;  // pixels > threshold are foreground
};

// Initialize labels: foreground = unique ID, background = 0
kernel void ccl_init(
    texture2d<float, access::read>   input   [[texture(0)]],
    device uint*                     labels  [[buffer(0)]],
    constant CCLParams&              params  [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;

    uint idx = gid.y * params.width + gid.x;
    float val = input.read(gid).r;

    if (val > params.threshold) {
        labels[idx] = idx + 1;  // 1-indexed unique label
    } else {
        labels[idx] = 0;        // background
    }
}

// One iteration: propagate minimum label from 4-connected neighbors
kernel void ccl_iterate(
    device uint*          labels_in  [[buffer(0)]],
    device uint*          labels_out [[buffer(1)]],
    device atomic_uint*   changed    [[buffer(2)]],
    constant CCLParams&   params     [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;

    uint idx = gid.y * params.width + gid.x;
    uint label = labels_in[idx];

    if (label == 0) {
        labels_out[idx] = 0;
        return;
    }

    uint min_label = label;
    int w = int(params.width);
    int h = int(params.height);
    int x = int(gid.x);
    int y = int(gid.y);

    // 4-connected neighbors
    if (x > 0) {
        uint n = labels_in[idx - 1];
        if (n > 0 && n < min_label) min_label = n;
    }
    if (x < w - 1) {
        uint n = labels_in[idx + 1];
        if (n > 0 && n < min_label) min_label = n;
    }
    if (y > 0) {
        uint n = labels_in[idx - params.width];
        if (n > 0 && n < min_label) min_label = n;
    }
    if (y < h - 1) {
        uint n = labels_in[idx + params.width];
        if (n > 0 && n < min_label) min_label = n;
    }

    labels_out[idx] = min_label;

    if (min_label != label) {
        atomic_fetch_add_explicit(changed, 1, memory_order_relaxed);
    }
}
