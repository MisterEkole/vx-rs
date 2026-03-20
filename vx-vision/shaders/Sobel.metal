#include <metal_stdlib>
using namespace metal;

// ================================================================
// Sobel / Scharr gradient kernels.
//
// sobel_3x3:  Compute Ix, Iy gradients using the 3x3 Sobel operator.
//             Outputs two R32Float textures (grad_x, grad_y).
//
// gradient_magnitude: Compute sqrt(Ix² + Iy²) and atan2(Iy, Ix).
//             Outputs magnitude (R32Float) and direction (R32Float).
//
// Dispatch as 2D grid: one thread per pixel.
// ================================================================

struct SobelParams {
    uint width;
    uint height;
};

kernel void sobel_3x3(
    texture2d<float, access::read>   input  [[texture(0)]],
    texture2d<float, access::write>  grad_x [[texture(1)]],
    texture2d<float, access::write>  grad_y [[texture(2)]],
    constant SobelParams&            params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;

    int w = int(params.width)  - 1;
    int h = int(params.height) - 1;
    int x = int(gid.x);
    int y = int(gid.y);

    // 3x3 neighborhood (clamped)
    float tl = input.read(uint2(clamp(x-1, 0, w), clamp(y-1, 0, h))).r;
    float tc = input.read(uint2(clamp(x,   0, w), clamp(y-1, 0, h))).r;
    float tr = input.read(uint2(clamp(x+1, 0, w), clamp(y-1, 0, h))).r;
    float ml = input.read(uint2(clamp(x-1, 0, w), clamp(y,   0, h))).r;
    float mr = input.read(uint2(clamp(x+1, 0, w), clamp(y,   0, h))).r;
    float bl = input.read(uint2(clamp(x-1, 0, w), clamp(y+1, 0, h))).r;
    float bc = input.read(uint2(clamp(x,   0, w), clamp(y+1, 0, h))).r;
    float br = input.read(uint2(clamp(x+1, 0, w), clamp(y+1, 0, h))).r;

    // Sobel Gx = [-1 0 +1; -2 0 +2; -1 0 +1]
    float gx = -tl + tr - 2.0*ml + 2.0*mr - bl + br;
    // Sobel Gy = [-1 -2 -1; 0 0 0; +1 +2 +1]
    float gy = -tl - 2.0*tc - tr + bl + 2.0*bc + br;

    grad_x.write(float4(gx, 0.0, 0.0, 1.0), gid);
    grad_y.write(float4(gy, 0.0, 0.0, 1.0), gid);
}

kernel void gradient_magnitude(
    texture2d<float, access::read>   grad_x    [[texture(0)]],
    texture2d<float, access::read>   grad_y    [[texture(1)]],
    texture2d<float, access::write>  magnitude [[texture(2)]],
    texture2d<float, access::write>  direction [[texture(3)]],
    constant SobelParams&            params    [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;

    float gx = grad_x.read(gid).r;
    float gy = grad_y.read(gid).r;

    float mag = sqrt(gx * gx + gy * gy);
    float dir = atan2(gy, gx);

    magnitude.write(float4(mag, 0.0, 0.0, 1.0), gid);
    direction.write(float4(dir, 0.0, 0.0, 1.0), gid);
}
