#include <metal_stdlib>
using namespace metal;

// ================================================================
// triangulate_midpoint — two-view 3D point triangulation.
// Dispatch: 1D per-match.
// ================================================================

struct TriangulateParams {
    uint  n_matches;
    uint  max_points;
    // Camera 1 intrinsics
    float fx1, fy1, cx1, cy1;
    // Camera 2 intrinsics
    float fx2, fy2, cx2, cy2;
    float _pad[2];  // align float4 to 16-byte boundary
    // Camera 1 pose (world-to-camera): 3 rows of float4
    float4 pose1_row0;
    float4 pose1_row1;
    float4 pose1_row2;
    // Camera 2 pose (world-to-camera): 3 rows of float4
    float4 pose2_row0;
    float4 pose2_row1;
    float4 pose2_row2;
};

struct Match2D {
    float u1, v1;  // pixel in image 1
    float u2, v2;  // pixel in image 2
};

struct Point3DOut {
    float x, y, z;
    float _pad;
};

// Invert a 3x4 [R|t] packed as 3 float4 rows → returns camera center in world space
// and inverse rotation rows
static void invert_pose(float4 r0, float4 r1, float4 r2,
                         thread float3& origin, thread float3& inv_r0, thread float3& inv_r1, thread float3& inv_r2)
{
    // R^T
    inv_r0 = float3(r0.x, r1.x, r2.x);
    inv_r1 = float3(r0.y, r1.y, r2.y);
    inv_r2 = float3(r0.z, r1.z, r2.z);
    // -R^T * t
    float3 t = float3(r0.w, r1.w, r2.w);
    origin = -float3(dot(inv_r0, t), dot(inv_r1, t), dot(inv_r2, t));
}

kernel void triangulate_midpoint(
    device Match2D*           matches   [[buffer(0)]],
    device Point3DOut*        points    [[buffer(1)]],
    device atomic_uint*       count     [[buffer(2)]],
    constant TriangulateParams& params  [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.n_matches) return;

    float u1 = matches[tid].u1;
    float v1 = matches[tid].v1;
    float u2 = matches[tid].u2;
    float v2 = matches[tid].v2;

    // Ray direction in camera frame
    float3 d1_cam = normalize(float3((u1 - params.cx1) / params.fx1, (v1 - params.cy1) / params.fy1, 1.0));
    float3 d2_cam = normalize(float3((u2 - params.cx2) / params.fx2, (v2 - params.cy2) / params.fy2, 1.0));

    // Transform rays to world frame
    float3 o1, ir0, ir1, ir2;
    invert_pose(params.pose1_row0, params.pose1_row1, params.pose1_row2, o1, ir0, ir1, ir2);
    float3 d1 = float3(dot(ir0, d1_cam), dot(ir1, d1_cam), dot(ir2, d1_cam));

    float3 o2, ir0b, ir1b, ir2b;
    invert_pose(params.pose2_row0, params.pose2_row1, params.pose2_row2, o2, ir0b, ir1b, ir2b);
    float3 d2 = float3(dot(ir0b, d2_cam), dot(ir1b, d2_cam), dot(ir2b, d2_cam));

    // Midpoint method: find closest points on two rays
    float3 w = o1 - o2;
    float a = dot(d1, d1);
    float b = dot(d1, d2);
    float c = dot(d2, d2);
    float d = dot(d1, w);
    float e = dot(d2, w);
    float denom = a * c - b * b;

    if (abs(denom) < 1e-10) return; // parallel rays

    float s = (b * e - c * d) / denom;
    float t = (a * e - b * d) / denom;

    // Reject if behind either camera
    if (s < 0.0 || t < 0.0) return;

    float3 p1 = o1 + d1 * s;
    float3 p2 = o2 + d2 * t;
    float3 midpoint = (p1 + p2) * 0.5;

    uint slot = atomic_fetch_add_explicit(count, 1, memory_order_relaxed);
    if (slot < params.max_points) {
        points[slot].x = midpoint.x;
        points[slot].y = midpoint.y;
        points[slot].z = midpoint.z;
        points[slot]._pad = 0.0;
    }
}
