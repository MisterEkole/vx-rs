#include <metal_stdlib>
using namespace metal;

// ================================================================
// Point cloud renderer — vertex + fragment shaders.
//
// Renders 3D points as colored circles (splats) with depth testing.
// ================================================================

struct PointRenderUniforms {
    float4x4 mvp;       // model-view-projection matrix
    float    point_size;
    float    _pad[3];
};

struct PointVertex {
    float x, y, z;
    float _pad0;
    uchar4 color;
    float nx, ny, nz;
    float _pad1;
};

struct VertexOut {
    float4 position [[position]];
    float4 color;
    float  point_size [[point_size]];
};

vertex VertexOut point_cloud_vertex(
    device PointVertex*        vertices  [[buffer(0)]],
    constant PointRenderUniforms& uniforms [[buffer(1)]],
    uint vid [[vertex_id]])
{
    VertexOut out;
    float4 pos = float4(vertices[vid].x, vertices[vid].y, vertices[vid].z, 1.0);
    out.position = uniforms.mvp * pos;
    out.color = float4(
        float(vertices[vid].color.r) / 255.0,
        float(vertices[vid].color.g) / 255.0,
        float(vertices[vid].color.b) / 255.0,
        1.0
    );
    out.point_size = uniforms.point_size;
    return out;
}

fragment float4 point_cloud_fragment(
    VertexOut in [[stage_in]],
    float2 point_coord [[point_coord]])
{
    // Circular point: discard outside radius
    float dist = length(point_coord - float2(0.5));
    if (dist > 0.5) discard_fragment();
    return in.color;
}
