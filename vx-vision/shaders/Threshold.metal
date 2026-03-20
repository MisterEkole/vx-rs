#include <metal_stdlib>
using namespace metal;

// ================================================================
// Thresholding operations.
//
// threshold_binary:   Global binary threshold.
// threshold_adaptive: Local adaptive threshold using a box filter
//                     over a window of (2*radius+1) pixels, computed
//                     from the integral image for O(1) per-pixel cost.
//
// Dispatch as 2D grid: one thread per pixel.
// ================================================================

struct ThresholdParams {
    uint  width;
    uint  height;
    float threshold;  // global threshold value (0.0–1.0 for normalized)
    int   invert;     // 0 = normal (above→1), 1 = inverted (above→0)
};

struct AdaptiveThresholdParams {
    uint  width;
    uint  height;
    int   radius;    // half-window size for local mean
    float c;         // constant subtracted from mean (bias)
    int   invert;
};

// Global binary threshold
kernel void threshold_binary(
    texture2d<float, access::read>   input  [[texture(0)]],
    texture2d<float, access::write>  output [[texture(1)]],
    constant ThresholdParams&        params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;

    float val = input.read(gid).r;
    float result;
    if (params.invert != 0) {
        result = (val > params.threshold) ? 0.0 : 1.0;
    } else {
        result = (val > params.threshold) ? 1.0 : 0.0;
    }
    output.write(float4(result, 0.0, 0.0, 1.0), gid);
}

// Adaptive threshold using precomputed integral image
// integral is the summed area table (R32Float)
kernel void threshold_adaptive(
    texture2d<float, access::read>   input    [[texture(0)]],
    texture2d<float, access::read>   integral [[texture(1)]],
    texture2d<float, access::write>  output   [[texture(2)]],
    constant AdaptiveThresholdParams& params  [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;

    int x = int(gid.x);
    int y = int(gid.y);
    int r = params.radius;

    // Window corners (clamped to image bounds)
    int x0 = max(x - r - 1, -1);
    int y0 = max(y - r - 1, -1);
    int x1 = min(x + r, int(params.width) - 1);
    int y1 = min(y + r, int(params.height) - 1);

    // Summed area table lookup
    float A = (x0 >= 0 && y0 >= 0) ? integral.read(uint2(x0, y0)).r : 0.0;
    float B = (y0 >= 0)            ? integral.read(uint2(x1, y0)).r : 0.0;
    float C = (x0 >= 0)            ? integral.read(uint2(x0, y1)).r : 0.0;
    float D = integral.read(uint2(x1, y1)).r;

    float area = float((x1 - x0) * (y1 - y0));
    float local_mean = (D - B - C + A) / area;

    float val = input.read(gid).r;
    float result;
    if (params.invert != 0) {
        result = (val > local_mean - params.c) ? 0.0 : 1.0;
    } else {
        result = (val > local_mean - params.c) ? 1.0 : 0.0;
    }
    output.write(float4(result, 0.0, 0.0, 1.0), gid);
}
