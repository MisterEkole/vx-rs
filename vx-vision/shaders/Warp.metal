#include <metal_stdlib>
using namespace metal;

// ================================================================
// Warp transforms: affine (2x3) and perspective (3x3).
//
// Both kernels compute the source coordinate for each output pixel
// using the INVERSE transform matrix, then bilinear-sample the input.
//
// Dispatch as 2D grid over the OUTPUT dimensions.
// ================================================================

struct WarpAffineParams {
    uint  width;
    uint  height;
    uint  src_width;
    uint  src_height;
    // Inverse affine matrix [a, b, tx, c, d, ty] (row-major 2x3)
    float m00, m01, m02;
    float m10, m11, m12;
};

struct WarpPerspectiveParams {
    uint  width;
    uint  height;
    uint  src_width;
    uint  src_height;
    // Inverse 3x3 homography (row-major)
    float m00, m01, m02;
    float m10, m11, m12;
    float m20, m21, m22;
};

static float bilinear_sample(texture2d<float, access::read> tex, float sx, float sy, int w, int h) {
    int x0 = int(floor(sx));
    int y0 = int(floor(sy));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    float fx = sx - float(x0);
    float fy = sy - float(y0);

    float tl = tex.read(uint2(clamp(x0, 0, w), clamp(y0, 0, h))).r;
    float tr = tex.read(uint2(clamp(x1, 0, w), clamp(y0, 0, h))).r;
    float bl = tex.read(uint2(clamp(x0, 0, w), clamp(y1, 0, h))).r;
    float br = tex.read(uint2(clamp(x1, 0, w), clamp(y1, 0, h))).r;

    return tl * (1.0-fx) * (1.0-fy) + tr * fx * (1.0-fy)
         + bl * (1.0-fx) * fy       + br * fx * fy;
}

kernel void warp_affine(
    texture2d<float, access::read>   input  [[texture(0)]],
    texture2d<float, access::write>  output [[texture(1)]],
    constant WarpAffineParams&       params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;

    float dx = float(gid.x);
    float dy = float(gid.y);

    // Apply inverse affine: src = M_inv * dst
    float sx = params.m00 * dx + params.m01 * dy + params.m02;
    float sy = params.m10 * dx + params.m11 * dy + params.m12;

    int w = int(params.src_width)  - 1;
    int h = int(params.src_height) - 1;

    // Out of bounds check
    if (sx < -1.0 || sx > float(w + 1) || sy < -1.0 || sy > float(h + 1)) {
        output.write(float4(0.0, 0.0, 0.0, 1.0), gid);
        return;
    }

    float val = bilinear_sample(input, sx, sy, w, h);
    output.write(float4(val, 0.0, 0.0, 1.0), gid);
}

kernel void warp_perspective(
    texture2d<float, access::read>   input  [[texture(0)]],
    texture2d<float, access::write>  output [[texture(1)]],
    constant WarpPerspectiveParams&  params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;

    float dx = float(gid.x);
    float dy = float(gid.y);

    // Apply inverse homography: src = H_inv * dst (homogeneous)
    float denom = params.m20 * dx + params.m21 * dy + params.m22;
    if (abs(denom) < 1e-8) {
        output.write(float4(0.0, 0.0, 0.0, 1.0), gid);
        return;
    }

    float sx = (params.m00 * dx + params.m01 * dy + params.m02) / denom;
    float sy = (params.m10 * dx + params.m11 * dy + params.m12) / denom;

    int w = int(params.src_width)  - 1;
    int h = int(params.src_height) - 1;

    if (sx < -1.0 || sx > float(w + 1) || sy < -1.0 || sy > float(h + 1)) {
        output.write(float4(0.0, 0.0, 0.0, 1.0), gid);
        return;
    }

    float val = bilinear_sample(input, sx, sy, w, h);
    output.write(float4(val, 0.0, 0.0, 1.0), gid);
}
