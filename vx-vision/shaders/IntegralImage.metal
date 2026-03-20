#include <metal_stdlib>
using namespace metal;

// ================================================================
// Integral image (summed area table) — two-pass sequential scan.
//
// Pass 1 (integral_rows): Prefix sum along each row.
//   Each thread processes one row sequentially.
//
// Pass 2 (integral_cols): Prefix sum along each column.
//   Each thread processes one column sequentially.
//
// After both passes: integral[y][x] = Σ input[j][i] for j≤y, i≤x.
//
// Uses R32Float textures for accumulation precision.
// ================================================================

struct IntegralParams {
    uint width;
    uint height;
};

// Pass 1: prefix sum per row (one thread per row)
kernel void integral_rows(
    texture2d<float, access::read>   input  [[texture(0)]],
    texture2d<float, access::write>  output [[texture(1)]],
    constant IntegralParams&         params [[buffer(0)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params.height) return;

    uint y = gid;
    float sum = 0.0;

    for (uint x = 0; x < params.width; x++) {
        sum += input.read(uint2(x, y)).r;
        output.write(float4(sum, 0.0, 0.0, 1.0), uint2(x, y));
    }
}

// Pass 2: prefix sum per column (one thread per column)
// Reads from the row-summed texture and writes cumulative column sums.
kernel void integral_cols(
    texture2d<float, access::read>   input  [[texture(0)]],
    texture2d<float, access::write>  output [[texture(1)]],
    constant IntegralParams&         params [[buffer(0)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params.width) return;

    uint x = gid;
    float sum = 0.0;

    for (uint y = 0; y < params.height; y++) {
        sum += input.read(uint2(x, y)).r;
        output.write(float4(sum, 0.0, 0.0, 1.0), uint2(x, y));
    }
}
