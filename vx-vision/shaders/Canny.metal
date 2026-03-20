#include <metal_stdlib>
using namespace metal;

// ================================================================
// Canny edge detection — two GPU passes after Sobel gradients.
//
// Pass 1 (canny_nms): Non-maximum suppression along gradient direction.
//   Reads magnitude + direction textures, outputs thinned edges.
//
// Pass 2 (canny_hysteresis): Double-threshold hysteresis.
//   Strong edges (>high) always kept. Weak edges (>low) kept only
//   if connected to a strong edge in the 3x3 neighborhood.
//   Multiple iterations may be needed for full propagation.
//
// Full pipeline: GaussianBlur → Sobel → canny_nms → canny_hysteresis
// ================================================================

struct CannyParams {
    uint  width;
    uint  height;
    float low_threshold;
    float high_threshold;
};

// Pass 1: NMS along gradient direction
kernel void canny_nms(
    texture2d<float, access::read>   magnitude  [[texture(0)]],
    texture2d<float, access::read>   direction  [[texture(1)]],
    texture2d<float, access::write>  output     [[texture(2)]],
    constant CannyParams&            params     [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;

    float mag = magnitude.read(gid).r;
    float dir = direction.read(gid).r;

    // Quantize direction to 4 axes: 0°, 45°, 90°, 135°
    // dir is in [-π, π], convert to [0, π) for symmetry
    float angle = dir < 0.0 ? dir + M_PI_F : dir;

    int w = int(params.width) - 1;
    int h = int(params.height) - 1;
    int x = int(gid.x);
    int y = int(gid.y);

    float n1, n2;

    if (angle < M_PI_F / 8.0 || angle >= 7.0 * M_PI_F / 8.0) {
        // 0° — horizontal edge, compare left/right
        n1 = magnitude.read(uint2(clamp(x-1, 0, w), y)).r;
        n2 = magnitude.read(uint2(clamp(x+1, 0, w), y)).r;
    } else if (angle < 3.0 * M_PI_F / 8.0) {
        // 45° — compare top-right / bottom-left
        n1 = magnitude.read(uint2(clamp(x+1, 0, w), clamp(y-1, 0, h))).r;
        n2 = magnitude.read(uint2(clamp(x-1, 0, w), clamp(y+1, 0, h))).r;
    } else if (angle < 5.0 * M_PI_F / 8.0) {
        // 90° — vertical edge, compare top/bottom
        n1 = magnitude.read(uint2(x, clamp(y-1, 0, h))).r;
        n2 = magnitude.read(uint2(x, clamp(y+1, 0, h))).r;
    } else {
        // 135° — compare top-left / bottom-right
        n1 = magnitude.read(uint2(clamp(x-1, 0, w), clamp(y-1, 0, h))).r;
        n2 = magnitude.read(uint2(clamp(x+1, 0, w), clamp(y+1, 0, h))).r;
    }

    // Suppress if not local maximum
    float result = (mag >= n1 && mag >= n2) ? mag : 0.0;
    output.write(float4(result, 0.0, 0.0, 1.0), gid);
}

// Pass 2: Hysteresis thresholding
// Output: 1.0 for edge, 0.0 for non-edge
kernel void canny_hysteresis(
    texture2d<float, access::read>   nms_output [[texture(0)]],
    texture2d<float, access::write>  edges      [[texture(1)]],
    constant CannyParams&            params     [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;

    float mag = nms_output.read(gid).r;

    // Strong edge: always keep
    if (mag >= params.high_threshold) {
        edges.write(float4(1.0, 0.0, 0.0, 1.0), gid);
        return;
    }

    // Below low threshold: always discard
    if (mag < params.low_threshold) {
        edges.write(float4(0.0, 0.0, 0.0, 1.0), gid);
        return;
    }

    // Weak edge: keep if any 3x3 neighbor is a strong edge
    int w = int(params.width) - 1;
    int h = int(params.height) - 1;
    int x = int(gid.x);
    int y = int(gid.y);

    bool has_strong_neighbor = false;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            int nx = clamp(x + dx, 0, w);
            int ny = clamp(y + dy, 0, h);
            if (nms_output.read(uint2(nx, ny)).r >= params.high_threshold) {
                has_strong_neighbor = true;
            }
        }
    }

    edges.write(float4(has_strong_neighbor ? 1.0 : 0.0, 0.0, 0.0, 1.0), gid);
}
