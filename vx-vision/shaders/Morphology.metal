#include <metal_stdlib>
using namespace metal;

// ================================================================
// Morphological operations: erode and dilate.
//
// Uses a rectangular structuring element defined by (radius_x, radius_y).
// Full kernel size is (2*radius_x+1) × (2*radius_y+1).
//
// erode:  output = min over neighborhood
// dilate: output = max over neighborhood
//
// Dispatch as 2D grid: one thread per pixel.
// ================================================================

struct MorphParams {
    uint width;
    uint height;
    int  radius_x;
    int  radius_y;
};

kernel void erode(
    texture2d<float, access::read>   input  [[texture(0)]],
    texture2d<float, access::write>  output [[texture(1)]],
    constant MorphParams&            params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;

    int w = int(params.width) - 1;
    int h = int(params.height) - 1;
    int cx = int(gid.x);
    int cy = int(gid.y);

    float min_val = 1.0;
    for (int dy = -params.radius_y; dy <= params.radius_y; dy++) {
        for (int dx = -params.radius_x; dx <= params.radius_x; dx++) {
            int sx = clamp(cx + dx, 0, w);
            int sy = clamp(cy + dy, 0, h);
            min_val = min(min_val, input.read(uint2(sx, sy)).r);
        }
    }

    output.write(float4(min_val, 0.0, 0.0, 1.0), gid);
}

kernel void dilate(
    texture2d<float, access::read>   input  [[texture(0)]],
    texture2d<float, access::write>  output [[texture(1)]],
    constant MorphParams&            params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;

    int w = int(params.width) - 1;
    int h = int(params.height) - 1;
    int cx = int(gid.x);
    int cy = int(gid.y);

    float max_val = 0.0;
    for (int dy = -params.radius_y; dy <= params.radius_y; dy++) {
        for (int dx = -params.radius_x; dx <= params.radius_x; dx++) {
            int sx = clamp(cx + dx, 0, w);
            int sy = clamp(cy + dy, 0, h);
            max_val = max(max_val, input.read(uint2(sx, sy)).r);
        }
    }

    output.write(float4(max_val, 0.0, 0.0, 1.0), gid);
}
