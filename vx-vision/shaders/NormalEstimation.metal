#include <metal_stdlib>
using namespace metal;

// ================================================================
// normal_estimate_organized — normals from depth via cross product.
// Dispatch: 2D per-pixel.
// ================================================================

struct NormalEstParams {
    float fx;
    float fy;
    float cx;
    float cy;
    uint  width;
    uint  height;
};

kernel void normal_estimate_organized(
    texture2d<float, access::read>   depth_in   [[texture(0)]],
    texture2d<float, access::write>  normal_out [[texture(1)]],
    constant NormalEstParams&        params     [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;
    if (gid.x == 0 || gid.y == 0 || gid.x >= params.width - 1 || gid.y >= params.height - 1) {
        normal_out.write(float4(0.0, 0.0, 0.0, 0.0), gid);
        return;
    }

    float d  = depth_in.read(gid).r;
    float dr = depth_in.read(uint2(gid.x + 1, gid.y)).r;
    float dd = depth_in.read(uint2(gid.x, gid.y + 1)).r;

    if (d <= 0.0 || dr <= 0.0 || dd <= 0.0) {
        normal_out.write(float4(0.0, 0.0, 0.0, 0.0), gid);
        return;
    }

    // Unproject 3 points
    float x = float(gid.x);
    float y = float(gid.y);

    float3 pc = float3((x - params.cx) * d / params.fx,
                        (y - params.cy) * d / params.fy,
                        d);
    float3 pr = float3((x + 1.0 - params.cx) * dr / params.fx,
                        (y - params.cy) * dr / params.fy,
                        dr);
    float3 pd = float3((x - params.cx) * dd / params.fx,
                        (y + 1.0 - params.cy) * dd / params.fy,
                        dd);

    float3 e1 = pr - pc;
    float3 e2 = pd - pc;
    float3 n = cross(e1, e2);

    float len = length(n);
    if (len > 1e-8) {
        n /= len;
        // Ensure normal points toward camera (negative Z in camera frame)
        if (n.z > 0.0) n = -n;
    } else {
        n = float3(0.0, 0.0, 0.0);
    }

    // Encode normal as RGB: (n + 1) / 2 mapped to [0, 1]
    normal_out.write(float4(n.x * 0.5 + 0.5, n.y * 0.5 + 0.5, n.z * 0.5 + 0.5, 1.0), gid);
}

// ================================================================
// normal_estimate_unorganized — normals via k-NN covariance.
// Dispatch: 1D per-point.
// ================================================================

struct NormalEstUnorgParams {
    uint  n_points;
    float radius;
    uint  max_neighbors;
};

struct PointXYZ {
    float x, y, z;
    float _pad;
};

kernel void normal_estimate_unorganized(
    device PointXYZ*             points     [[buffer(0)]],
    device float*                normals    [[buffer(1)]],  // n_points × 3
    constant NormalEstUnorgParams& params   [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.n_points) return;

    float3 center = float3(points[tid].x, points[tid].y, points[tid].z);
    float radius2 = params.radius * params.radius;

    // Accumulate covariance matrix from neighbors
    float3 mean = float3(0.0);
    int count = 0;

    // Pass 1: compute mean of neighbors
    for (uint i = 0; i < params.n_points && count < int(params.max_neighbors); i++) {
        float3 p = float3(points[i].x, points[i].y, points[i].z);
        float3 diff = p - center;
        float d2 = dot(diff, diff);
        if (d2 < radius2 && d2 > 1e-10) {
            mean += p;
            count++;
        }
    }

    if (count < 3) {
        normals[tid * 3 + 0] = 0.0;
        normals[tid * 3 + 1] = 0.0;
        normals[tid * 3 + 2] = 0.0;
        return;
    }

    mean /= float(count);

    // Pass 2: compute covariance matrix
    float cov00 = 0.0, cov01 = 0.0, cov02 = 0.0;
    float cov11 = 0.0, cov12 = 0.0, cov22 = 0.0;

    for (uint i = 0; i < params.n_points; i++) {
        float3 p = float3(points[i].x, points[i].y, points[i].z);
        float3 diff = p - center;
        float d2 = dot(diff, diff);
        if (d2 < radius2 && d2 > 1e-10) {
            float3 d = p - mean;
            cov00 += d.x * d.x;
            cov01 += d.x * d.y;
            cov02 += d.x * d.z;
            cov11 += d.y * d.y;
            cov12 += d.y * d.z;
            cov22 += d.z * d.z;
        }
    }

    // Analytic smallest eigenvector of 3x3 symmetric matrix
    // Using the cross-product method: the normal is the cross product
    // of two rows of (C - lambda_min * I)
    // For simplicity, use power iteration to find largest eigenvector,
    // then the normal is perpendicular to it.
    // Actually, simplest robust approach: cross product of two column vectors
    // formed from covariance rows.
    float3 row0 = float3(cov00, cov01, cov02);
    float3 row1 = float3(cov01, cov11, cov12);
    float3 row2 = float3(cov02, cov12, cov22);

    // The normal is the eigenvector with smallest eigenvalue.
    // For a plane, two eigenvalues are large and one is small.
    // Cross product of the two rows with largest magnitude gives the normal direction.
    float3 c01 = cross(row0, row1);
    float3 c02 = cross(row0, row2);
    float3 c12 = cross(row1, row2);

    float l01 = dot(c01, c01);
    float l02 = dot(c02, c02);
    float l12 = dot(c12, c12);

    float3 n;
    if (l01 >= l02 && l01 >= l12) {
        n = c01;
    } else if (l02 >= l12) {
        n = c02;
    } else {
        n = c12;
    }

    float len = length(n);
    if (len > 1e-8) {
        n /= len;
    } else {
        n = float3(0.0, 0.0, 1.0);
    }

    normals[tid * 3 + 0] = n.x;
    normals[tid * 3 + 1] = n.y;
    normals[tid * 3 + 2] = n.z;
}
