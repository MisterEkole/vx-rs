#include <metal_stdlib>
using namespace metal;

// ================================================================
// Mesh renderer — vertex + fragment shaders with Phong shading.
// ================================================================

struct MeshRenderUniforms {
    float4x4 mvp;        // model-view-projection
    float4x4 model;      // model matrix for normal transform
    float4   light_dir;  // world-space light direction (normalized)
    float4   light_color;
    float4   ambient;
};

struct MeshVertex {
    float3 position;
    float3 normal;
    float2 uv;
};

struct MeshVertexOut {
    float4 position [[position]];
    float3 normal;
    float3 world_pos;
};

vertex MeshVertexOut mesh_vertex(
    device MeshVertex*          vertices [[buffer(0)]],
    constant MeshRenderUniforms& uniforms [[buffer(1)]],
    uint vid [[vertex_id]])
{
    MeshVertexOut out;
    float4 pos = float4(vertices[vid].position, 1.0);
    out.position = uniforms.mvp * pos;
    out.world_pos = (uniforms.model * pos).xyz;
    // Transform normal by inverse transpose of model matrix (approximated for rigid transforms)
    out.normal = normalize((uniforms.model * float4(vertices[vid].normal, 0.0)).xyz);
    return out;
}

fragment float4 mesh_fragment(
    MeshVertexOut in [[stage_in]],
    constant MeshRenderUniforms& uniforms [[buffer(1)]])
{
    float3 n = normalize(in.normal);
    float3 l = normalize(uniforms.light_dir.xyz);

    // Lambertian diffuse
    float ndotl = max(dot(n, l), 0.0);
    float3 diffuse = uniforms.light_color.rgb * ndotl;
    float3 color = uniforms.ambient.rgb + diffuse;

    return float4(clamp(color, 0.0, 1.0), 1.0);
}
