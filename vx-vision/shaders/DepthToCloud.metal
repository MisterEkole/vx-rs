#include <metal_stdlib>
using namespace metal;

// ================================================================
// depth_to_cloud — unproject depth to 3D points with optional color.
// Dispatch: 2D per-pixel.
// ================================================================

struct GpuPoint3D {
    float px, py, pz;   // position (3 floats, no float3 alignment issues)
    float _pad0;
    uchar4 color;
    float nx, ny, nz;   // normal
    float _pad1;
};

struct DepthToCloudParams {
    float fx;
    float fy;
    float cx;
    float cy;
    float min_depth;
    float max_depth;
    float depth_scale;
    uint  width;
    uint  height;
    uint  max_points;
};

kernel void depth_to_cloud(
    texture2d<float, access::read>  depth_tex  [[texture(0)]],
    texture2d<float, access::read>  color_tex  [[texture(1)]],
    device GpuPoint3D*              points     [[buffer(0)]],
    device atomic_uint*             count      [[buffer(1)]],
    constant DepthToCloudParams&    params     [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;

    float d = depth_tex.read(gid).r * params.depth_scale;

    if (d < params.min_depth || d > params.max_depth) return;

    // Unproject to 3D camera coordinates
    float X = (float(gid.x) - params.cx) * d / params.fx;
    float Y = (float(gid.y) - params.cy) * d / params.fy;
    float Z = d;

    // Sample color if available (check if color texture has non-zero size)
    uchar4 color = uchar4(255, 255, 255, 255);
    if (color_tex.get_width() > 1) {
        float4 c = color_tex.read(gid);
        color = uchar4(uchar(c.r * 255.0), uchar(c.g * 255.0), uchar(c.b * 255.0), 255);
    }

    uint slot = atomic_fetch_add_explicit(count, 1, memory_order_relaxed);
    if (slot < params.max_points) {
        points[slot].px = X;
        points[slot].py = Y;
        points[slot].pz = Z;
        points[slot]._pad0 = 0.0;
        points[slot].color = color;
        points[slot].nx = 0.0;
        points[slot].ny = 0.0;
        points[slot].nz = 0.0;
        points[slot]._pad1 = 0.0;
    }
}
