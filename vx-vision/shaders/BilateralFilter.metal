#include <metal_stdlib>
using namespace metal;

// ================================================================
// Bilateral filter — edge-preserving smoothing.
//
// Unlike Gaussian blur, weights depend on BOTH spatial distance
// AND intensity difference. Edges (large intensity jumps) are
// preserved because distant-intensity neighbors get low weight.
//
// Dispatch as 2D grid: one thread per pixel.
// ================================================================

struct BilateralParams {
    uint  width;
    uint  height;
    int   radius;         // spatial window half-size
    float sigma_spatial;  // spatial Gaussian sigma
    float sigma_range;    // intensity Gaussian sigma
};

kernel void bilateral_filter(
    texture2d<float, access::read>   input  [[texture(0)]],
    texture2d<float, access::write>  output [[texture(1)]],
    constant BilateralParams&        params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;

    float center = input.read(gid).r;
    float inv_2ss = 1.0 / (2.0 * params.sigma_spatial * params.sigma_spatial);
    float inv_2sr = 1.0 / (2.0 * params.sigma_range * params.sigma_range);

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

            float neighbor = input.read(uint2(sx, sy)).r;

            // Spatial weight
            float spatial_dist2 = float(dx * dx + dy * dy);
            float ws = exp(-spatial_dist2 * inv_2ss);

            // Range (intensity) weight
            float range_diff = neighbor - center;
            float wr = exp(-range_diff * range_diff * inv_2sr);

            float weight = ws * wr;
            sum += neighbor * weight;
            weight_sum += weight;
        }
    }

    float result = sum / max(weight_sum, 1e-8);
    output.write(float4(result, 0.0, 0.0, 1.0), gid);
}
