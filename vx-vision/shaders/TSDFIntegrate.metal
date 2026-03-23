#include <metal_stdlib>
using namespace metal;

// ================================================================
// tsdf_integrate — fuse a depth frame into the TSDF voxel grid.
// Dispatch: 2D over (res_x, res_y) with inner Z loop.
// ================================================================

struct TSDFIntegrateParams {
    uint  res_x;
    uint  res_y;
    uint  res_z;
    float voxel_size;
    float origin_x;
    float origin_y;
    float origin_z;
    float truncation_dist;
    float max_weight;
    // Camera intrinsics
    float fx;
    float fy;
    float cx;
    float cy;
    uint  img_width;
    uint  img_height;
    // Camera extrinsics packed as 3 rows of float4: [r0 r1 r2 t]
    float4 pose_row0;
    float4 pose_row1;
    float4 pose_row2;
};

kernel void tsdf_integrate(
    texture2d<float, access::read>  depth_tex  [[texture(0)]],
    device float*                   tsdf       [[buffer(0)]],
    device float*                   weights    [[buffer(1)]],
    constant TSDFIntegrateParams&   params     [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= params.res_x || gid.y >= params.res_y) return;

    uint ix = gid.x;
    uint iy = gid.y;

    for (uint iz = 0; iz < params.res_z; iz++) {
        // Voxel center in world space
        float wx = params.origin_x + (float(ix) + 0.5) * params.voxel_size;
        float wy = params.origin_y + (float(iy) + 0.5) * params.voxel_size;
        float wz = params.origin_z + (float(iz) + 0.5) * params.voxel_size;

        // Transform world → camera: p_cam = R * p_world + t
        float cx = params.pose_row0.x * wx + params.pose_row0.y * wy + params.pose_row0.z * wz + params.pose_row0.w;
        float cy = params.pose_row1.x * wx + params.pose_row1.y * wy + params.pose_row1.z * wz + params.pose_row1.w;
        float cz = params.pose_row2.x * wx + params.pose_row2.y * wy + params.pose_row2.z * wz + params.pose_row2.w;

        // Skip voxels behind camera
        if (cz <= 0.0) continue;

        // Project to image
        float u = params.fx * cx / cz + params.cx;
        float v = params.fy * cy / cz + params.cy;

        // Bounds check
        int iu = int(round(u));
        int iv = int(round(v));
        if (iu < 0 || iu >= int(params.img_width) || iv < 0 || iv >= int(params.img_height)) continue;

        // Read depth at this pixel
        float depth = depth_tex.read(uint2(iu, iv)).r;
        if (depth <= 0.0) continue;

        // Signed distance: positive = in front of surface, negative = behind
        float sdf = depth - cz;

        // Skip voxels outside truncation band
        if (sdf < -params.truncation_dist) continue;

        // Truncate
        float tsdf_val = clamp(sdf / params.truncation_dist, -1.0, 1.0);

        // Update running weighted average
        uint idx = (iz * params.res_y + iy) * params.res_x + ix;
        float old_tsdf = tsdf[idx];
        float old_weight = weights[idx];
        float new_weight = min(old_weight + 1.0, params.max_weight);

        tsdf[idx] = (old_tsdf * old_weight + tsdf_val) / new_weight;
        weights[idx] = new_weight;
    }
}
