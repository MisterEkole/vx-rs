#include <metal_stdlib>
using namespace metal;

// ================================================================
// depth_inpaint — iterative nearest-valid-neighbor hole fill.
// Dispatch: 2D per-pixel per iteration.
// ================================================================

struct DepthInpaintParams {
    uint width;
    uint height;
    int  step_size;  // current search radius (doubles each iteration)
};

kernel void depth_inpaint(
    texture2d<float, access::read>   input  [[texture(0)]],
    texture2d<float, access::write>  output [[texture(1)]],
    constant DepthInpaintParams&     params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;

    float center = input.read(gid).r;

    // If pixel already has valid depth, pass through
    if (center > 0.0) {
        output.write(float4(center, 0.0, 0.0, 1.0), gid);
        return;
    }

    // Search the 8 neighbours at current step distance
    int step = params.step_size;
    int cx = int(gid.x);
    int cy = int(gid.y);
    int w = int(params.width) - 1;
    int h = int(params.height) - 1;

    float best = 0.0;
    float best_dist2 = 1e10;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;

            int sx = clamp(cx + dx * step, 0, w);
            int sy = clamp(cy + dy * step, 0, h);
            float d = input.read(uint2(sx, sy)).r;

            if (d > 0.0) {
                float dist2 = float(dx * dx + dy * dy);
                if (dist2 < best_dist2) {
                    best_dist2 = dist2;
                    best = d;
                }
            }
        }
    }

    output.write(float4(best, 0.0, 0.0, 1.0), gid);
}
