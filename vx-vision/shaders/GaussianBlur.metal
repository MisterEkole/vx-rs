#include <metal_stdlib>
using namespace metal;

struct GaussianParams {
    uint  width;
    uint  height;
    float sigma;
    uint  radius;   // kernel half-width: kernel spans [-radius, +radius]
};

// ================================================================
// Separable Gaussian blur — two-pass pipeline.
//
// Pass 1 (gaussian_blur_h): horizontal 1D convolution → intermediate
// Pass 2 (gaussian_blur_v): vertical   1D convolution → final output
//
// Both dispatched as 2D grids (one thread per pixel).
// Kernel weights are computed from the Gaussian formula inline;
// clamp_to_edge is applied at boundaries.
// ================================================================

kernel void gaussian_blur_h(
    texture2d<float, access::read>   input  [[texture(0)]],
    texture2d<float, access::write>  output [[texture(1)]],
    constant GaussianParams&         params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;

    float sum        = 0.0;
    float weight_sum = 0.0;
    float inv_two_s2 = 1.0 / (2.0 * params.sigma * params.sigma);
    int   r          = int(params.radius);

    for (int dx = -r; dx <= r; dx++) {
        int   sx = clamp(int(gid.x) + dx, 0, int(params.width) - 1);
        float w  = exp(-float(dx * dx) * inv_two_s2);
        sum        += input.read(uint2(sx, gid.y)).r * w;
        weight_sum += w;
    }

    output.write(float4(sum / weight_sum, 0.0, 0.0, 1.0), gid);
}

kernel void gaussian_blur_v(
    texture2d<float, access::read>   input  [[texture(0)]],
    texture2d<float, access::write>  output [[texture(1)]],
    constant GaussianParams&         params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;

    float sum        = 0.0;
    float weight_sum = 0.0;
    float inv_two_s2 = 1.0 / (2.0 * params.sigma * params.sigma);
    int   r          = int(params.radius);

    for (int dy = -r; dy <= r; dy++) {
        int   sy = clamp(int(gid.y) + dy, 0, int(params.height) - 1);
        float w  = exp(-float(dy * dy) * inv_two_s2);
        sum        += input.read(uint2(gid.x, sy)).r * w;
        weight_sum += w;
    }

    output.write(float4(sum / weight_sum, 0.0, 0.0, 1.0), gid);
}
