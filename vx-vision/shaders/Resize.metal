#include <metal_stdlib>
using namespace metal;

// ================================================================
// Image resize with bilinear interpolation.
//
// Dispatch as 2D grid over the OUTPUT dimensions.
// One thread per output pixel.
// ================================================================

struct ResizeParams {
    uint src_width;
    uint src_height;
    uint dst_width;
    uint dst_height;
};

kernel void resize_bilinear(
    texture2d<float, access::read>   input  [[texture(0)]],
    texture2d<float, access::write>  output [[texture(1)]],
    constant ResizeParams&           params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= params.dst_width || gid.y >= params.dst_height) return;

    // Map output pixel to input coordinate (center-aligned)
    float sx = (float(gid.x) + 0.5) * float(params.src_width)  / float(params.dst_width)  - 0.5;
    float sy = (float(gid.y) + 0.5) * float(params.src_height) / float(params.dst_height) - 0.5;

    int x0 = int(floor(sx));
    int y0 = int(floor(sy));
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float fx = sx - float(x0);
    float fy = sy - float(y0);

    int w = int(params.src_width)  - 1;
    int h = int(params.src_height) - 1;

    float tl = input.read(uint2(clamp(x0, 0, w), clamp(y0, 0, h))).r;
    float tr = input.read(uint2(clamp(x1, 0, w), clamp(y0, 0, h))).r;
    float bl = input.read(uint2(clamp(x0, 0, w), clamp(y1, 0, h))).r;
    float br = input.read(uint2(clamp(x1, 0, w), clamp(y1, 0, h))).r;

    float val = tl * (1.0 - fx) * (1.0 - fy)
              + tr * fx * (1.0 - fy)
              + bl * (1.0 - fx) * fy
              + br * fx * fy;

    output.write(float4(val, 0.0, 0.0, 1.0), gid);
}
