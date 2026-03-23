#include <metal_stdlib>
using namespace metal;

// ================================================================
// tsdf_raycast — raycast TSDF volume, output depth + normals.
// Dispatch: 2D per-pixel.
// ================================================================

struct TSDFRaycastParams {
    uint  res_x;
    uint  res_y;
    uint  res_z;
    float voxel_size;
    float origin_x;
    float origin_y;
    float origin_z;
    float truncation_dist;
    // Camera intrinsics
    float fx;
    float fy;
    float cx;
    float cy;
    uint  img_width;
    uint  img_height;
    // Inverse camera extrinsics (camera-to-world): 3 rows of float4
    float4 inv_pose_row0;
    float4 inv_pose_row1;
    float4 inv_pose_row2;
};

// Trilinear interpolation of TSDF
static float sample_tsdf(device float* tsdf, uint rx, uint ry, uint rz, float3 pos, float3 origin, float voxel_size) {
    float3 grid_pos = (pos - origin) / voxel_size - 0.5;
    int3 base = int3(floor(grid_pos));
    float3 frac = grid_pos - float3(base);

    // Clamp to grid bounds
    int3 b0 = clamp(base, int3(0), int3(rx - 2, ry - 2, rz - 2));
    int3 b1 = b0 + 1;

    // 8 corner values
    float c000 = tsdf[(b0.z * ry + b0.y) * rx + b0.x];
    float c100 = tsdf[(b0.z * ry + b0.y) * rx + b1.x];
    float c010 = tsdf[(b0.z * ry + b1.y) * rx + b0.x];
    float c110 = tsdf[(b0.z * ry + b1.y) * rx + b1.x];
    float c001 = tsdf[(b1.z * ry + b0.y) * rx + b0.x];
    float c101 = tsdf[(b1.z * ry + b0.y) * rx + b1.x];
    float c011 = tsdf[(b1.z * ry + b1.y) * rx + b0.x];
    float c111 = tsdf[(b1.z * ry + b1.y) * rx + b1.x];

    // Trilinear interpolation
    float c00 = mix(c000, c100, frac.x);
    float c10 = mix(c010, c110, frac.x);
    float c01 = mix(c001, c101, frac.x);
    float c11 = mix(c011, c111, frac.x);

    float c0 = mix(c00, c10, frac.y);
    float c1 = mix(c01, c11, frac.y);

    return mix(c0, c1, frac.z);
}

kernel void tsdf_raycast(
    device float*                    tsdf       [[buffer(0)]],
    texture2d<float, access::write>  depth_out  [[texture(0)]],
    texture2d<float, access::write>  normal_out [[texture(1)]],
    constant TSDFRaycastParams&      params     [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= params.img_width || gid.y >= params.img_height) return;

    // Ray origin in camera space (camera center = 0)
    // Ray direction in camera space
    float3 ray_dir_cam = normalize(float3(
        (float(gid.x) - params.cx) / params.fx,
        (float(gid.y) - params.cy) / params.fy,
        1.0
    ));

    // Transform ray to world space using inverse pose
    float3 ray_origin = float3(params.inv_pose_row0.w, params.inv_pose_row1.w, params.inv_pose_row2.w);
    float3 ray_dir = float3(
        params.inv_pose_row0.x * ray_dir_cam.x + params.inv_pose_row0.y * ray_dir_cam.y + params.inv_pose_row0.z * ray_dir_cam.z,
        params.inv_pose_row1.x * ray_dir_cam.x + params.inv_pose_row1.y * ray_dir_cam.y + params.inv_pose_row1.z * ray_dir_cam.z,
        params.inv_pose_row2.x * ray_dir_cam.x + params.inv_pose_row2.y * ray_dir_cam.y + params.inv_pose_row2.z * ray_dir_cam.z
    );
    ray_dir = normalize(ray_dir);

    // Volume bounds
    float3 vol_min = float3(params.origin_x, params.origin_y, params.origin_z);
    float3 vol_max = vol_min + float3(float(params.res_x), float(params.res_y), float(params.res_z)) * params.voxel_size;

    // Ray-box intersection to find entry/exit
    float3 inv_dir = 1.0 / ray_dir;
    float3 t0 = (vol_min - ray_origin) * inv_dir;
    float3 t1 = (vol_max - ray_origin) * inv_dir;
    float3 tmin_v = min(t0, t1);
    float3 tmax_v = max(t0, t1);

    float t_enter = max(max(tmin_v.x, tmin_v.y), max(tmin_v.z, 0.001));
    float t_exit  = min(min(tmax_v.x, tmax_v.y), tmax_v.z);

    if (t_enter >= t_exit) {
        depth_out.write(float4(0.0), gid);
        normal_out.write(float4(0.5, 0.5, 0.5, 0.0), gid);
        return;
    }

    // March through volume
    float step_size = params.voxel_size * 0.5;
    float prev_tsdf = 1.0;
    float prev_t = t_enter;
    float found_depth = 0.0;
    float3 found_pos = float3(0.0);
    bool found = false;

    for (float t = t_enter; t < t_exit; t += step_size) {
        float3 pos = ray_origin + ray_dir * t;
        float tsdf_val = sample_tsdf(tsdf, params.res_x, params.res_y, params.res_z,
                                      pos, vol_min, params.voxel_size);

        // Zero crossing: prev > 0, current < 0
        if (prev_tsdf > 0.0 && tsdf_val < 0.0) {
            // Linear interpolation to find exact crossing
            float t_cross = prev_t + step_size * prev_tsdf / (prev_tsdf - tsdf_val);
            found_pos = ray_origin + ray_dir * t_cross;
            found_depth = t_cross;
            found = true;
            break;
        }

        prev_tsdf = tsdf_val;
        prev_t = t;
    }

    if (!found) {
        depth_out.write(float4(0.0), gid);
        normal_out.write(float4(0.5, 0.5, 0.5, 0.0), gid);
        return;
    }

    // Compute normal via central differences
    float delta = params.voxel_size;
    float dx = sample_tsdf(tsdf, params.res_x, params.res_y, params.res_z, found_pos + float3(delta, 0, 0), vol_min, params.voxel_size)
             - sample_tsdf(tsdf, params.res_x, params.res_y, params.res_z, found_pos - float3(delta, 0, 0), vol_min, params.voxel_size);
    float dy = sample_tsdf(tsdf, params.res_x, params.res_y, params.res_z, found_pos + float3(0, delta, 0), vol_min, params.voxel_size)
             - sample_tsdf(tsdf, params.res_x, params.res_y, params.res_z, found_pos - float3(0, delta, 0), vol_min, params.voxel_size);
    float dz = sample_tsdf(tsdf, params.res_x, params.res_y, params.res_z, found_pos + float3(0, 0, delta), vol_min, params.voxel_size)
             - sample_tsdf(tsdf, params.res_x, params.res_y, params.res_z, found_pos - float3(0, 0, delta), vol_min, params.voxel_size);

    float3 normal = normalize(float3(dx, dy, dz));

    depth_out.write(float4(found_depth, 0.0, 0.0, 1.0), gid);
    normal_out.write(float4(normal * 0.5 + 0.5, 1.0), gid);
}
