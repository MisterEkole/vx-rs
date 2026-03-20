#include <metal_stdlib>
using namespace metal;

// ================================================================
// GPU image pyramid — 2x Gaussian downsample.
//
// Applies a 5-tap Gaussian filter [1, 4, 6, 4, 1] / 16 in both
// directions then samples every other pixel. Output is half the
// input resolution in each dimension.
//
// Dispatch as 2D grid over the OUTPUT dimensions.
// ================================================================

struct PyramidParams {
    uint src_width;
    uint src_height;
    uint dst_width;
    uint dst_height;
};

kernel void pyramid_downsample(
    texture2d<float, access::read>   input  [[texture(0)]],
    texture2d<float, access::write>  output [[texture(1)]],
    constant PyramidParams&          params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= params.dst_width || gid.y >= params.dst_height) return;

    // Source center pixel (2x output coordinate)
    int cx = int(gid.x) * 2;
    int cy = int(gid.y) * 2;

    int w = int(params.src_width);
    int h = int(params.src_height);

    // 5-tap Gaussian weights: [1, 4, 6, 4, 1] / 256 (separable)
    const float k[5] = { 1.0/16.0, 4.0/16.0, 6.0/16.0, 4.0/16.0, 1.0/16.0 };

    float sum = 0.0;
    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            int sx = clamp(cx + dx, 0, w - 1);
            int sy = clamp(cy + dy, 0, h - 1);
            float weight = k[dx + 2] * k[dy + 2];
            sum += input.read(uint2(sx, sy)).r * weight;
        }
    }

    output.write(float4(sum, 0.0, 0.0, 1.0), gid);
}
