#include <metal_stdlib>
using namespace metal;

// ================================================================
// Histogram computation and equalization.
//
// histogram_compute:   Atomic 256-bin histogram of a grayscale image.
//                      Dispatch as 2D grid over image dimensions.
//
// histogram_equalize:  Per-pixel LUT remap using a precomputed CDF.
//                      The CDF (256 floats, normalized to [0,1]) is
//                      computed on the CPU from the histogram.
//                      Dispatch as 2D grid over image dimensions.
//
// ================================================================

struct HistogramParams {
    uint width;
    uint height;
};

// Compute 256-bin histogram using atomic increments
// Output buffer: 256 × uint, zero-initialized before dispatch.
kernel void histogram_compute(
    texture2d<float, access::read>        input  [[texture(0)]],
    device atomic_uint*                   bins   [[buffer(0)]],
    constant HistogramParams&             params [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;

    float val = input.read(gid).r;
    uint bin = min(uint(val * 255.0), 255u);
    atomic_fetch_add_explicit(&bins[bin], 1, memory_order_relaxed);
}

// Equalize using precomputed CDF lookup table
// cdf buffer: 256 × float, normalized to [0, 1]
kernel void histogram_equalize(
    texture2d<float, access::read>   input  [[texture(0)]],
    texture2d<float, access::write>  output [[texture(1)]],
    device float*                    cdf    [[buffer(0)]],
    constant HistogramParams&        params [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;

    float val = input.read(gid).r;
    uint bin = min(uint(val * 255.0), 255u);
    float equalized = cdf[bin];
    output.write(float4(equalized, 0.0, 0.0, 1.0), gid);
}
