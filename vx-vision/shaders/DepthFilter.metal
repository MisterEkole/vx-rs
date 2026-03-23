#include <metal_stdlib>
using namespace metal;

// ================================================================
// depth_bilateral — edge-preserving depth smoothing.
// Dispatch: 2D per-pixel.
// ================================================================

struct DepthBilateralParams {
    uint  width;
    uint  height;
    int   radius;
    float sigma_spatial;
    float sigma_depth;
};

kernel void depth_bilateral(
    texture2d<float, access::read>   depth_in  [[texture(0)]],
    texture2d<float, access::write>  depth_out [[texture(1)]],
    constant DepthBilateralParams&   params    [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;

    float center = depth_in.read(gid).r;

    // Skip invalid pixels
    if (center <= 0.0) {
        depth_out.write(float4(0.0, 0.0, 0.0, 1.0), gid);
        return;
    }

    float inv_2ss = 1.0 / (2.0 * params.sigma_spatial * params.sigma_spatial);
    float inv_2sd = 1.0 / (2.0 * params.sigma_depth * params.sigma_depth);

    int w = int(params.width) - 1;
    int h = int(params.height) - 1;
    int cx = int(gid.x);
    int cy = int(gid.y);

    float sum = 0.0;
    float weight_sum = 0.0;

    for (int dy = -params.radius; dy <= params.radius; dy++) {
        for (int dx = -params.radius; dx <= params.radius; dx++) {
            int sx = clamp(cx + dx, 0, w);
            int sy = clamp(cy + dy, 0, h);

            float neighbor = depth_in.read(uint2(sx, sy)).r;
            if (neighbor <= 0.0) continue; // skip invalid

            float spatial_dist2 = float(dx * dx + dy * dy);
            float ws = exp(-spatial_dist2 * inv_2ss);

            float depth_diff = neighbor - center;
            float wd = exp(-depth_diff * depth_diff * inv_2sd);

            float weight = ws * wd;
            sum += neighbor * weight;
            weight_sum += weight;
        }
    }

    float result = (weight_sum > 1e-8) ? (sum / weight_sum) : center;
    depth_out.write(float4(result, 0.0, 0.0, 1.0), gid);
}

// ================================================================
// depth_median — median filter for depth maps.
// Dispatch: 2D per-pixel.
// ================================================================

struct DepthMedianParams {
    uint width;
    uint height;
    int  radius;
};

kernel void depth_median(
    texture2d<float, access::read>   depth_in  [[texture(0)]],
    texture2d<float, access::write>  depth_out [[texture(1)]],
    constant DepthMedianParams&      params    [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;

    int w = int(params.width) - 1;
    int h = int(params.height) - 1;
    int cx = int(gid.x);
    int cy = int(gid.y);

    // Collect valid depth values in a small fixed-size array
    float vals[121]; // max (2*5+1)^2 = 121
    int count = 0;

    for (int dy = -params.radius; dy <= params.radius; dy++) {
        for (int dx = -params.radius; dx <= params.radius; dx++) {
            int sx = clamp(cx + dx, 0, w);
            int sy = clamp(cy + dy, 0, h);
            float d = depth_in.read(uint2(sx, sy)).r;
            if (d > 0.0 && count < 121) {
                vals[count++] = d;
            }
        }
    }

    // Simple bubble-sort for small arrays (max 25 elements typical)
    for (int i = 0; i < count - 1; i++) {
        for (int j = 0; j < count - 1 - i; j++) {
            if (vals[j] > vals[j + 1]) {
                float tmp = vals[j];
                vals[j] = vals[j + 1];
                vals[j + 1] = tmp;
            }
        }
    }

    float result = (count > 0) ? vals[count / 2] : 0.0;
    depth_out.write(float4(result, 0.0, 0.0, 1.0), gid);
}
