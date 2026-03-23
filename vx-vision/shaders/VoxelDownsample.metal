#include <metal_stdlib>
using namespace metal;

// ================================================================
// Voxel grid downsampling for point clouds.
// Pass 1: hash + accumulate. Pass 2: average + compact.
// Dispatch: 1D per-point (pass 1), 1D per-cell (pass 2).
// ================================================================

struct VoxelDownsampleParams {
    uint  n_points;
    float voxel_size;
    float origin_x;
    float origin_y;
    float origin_z;
    uint  table_size;    // hash table size (power of 2)
};

struct PointXYZ {
    float x, y, z;
    float _pad;
};

// Hash function for 3D grid coordinates
static uint hash_coord(int ix, int iy, int iz, uint table_size) {
    uint h = uint(ix) * 73856093u ^ uint(iy) * 19349669u ^ uint(iz) * 83492791u;
    return h % table_size;
}

// Pass 1: accumulate points into voxel cells via fixed-point atomics
kernel void voxel_hash_assign(
    device PointXYZ*               points      [[buffer(0)]],
    device atomic_uint*            cell_counts [[buffer(1)]],  // table_size
    device atomic_uint*            cell_sum_x  [[buffer(2)]],  // table_size (fixed-point)
    device atomic_uint*            cell_sum_y  [[buffer(3)]],
    device atomic_uint*            cell_sum_z  [[buffer(4)]],
    constant VoxelDownsampleParams& params     [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.n_points) return;

    float px = points[tid].x;
    float py = points[tid].y;
    float pz = points[tid].z;

    int ix = int(floor((px - params.origin_x) / params.voxel_size));
    int iy = int(floor((py - params.origin_y) / params.voxel_size));
    int iz = int(floor((pz - params.origin_z) / params.voxel_size));

    uint h = hash_coord(ix, iy, iz, params.table_size);

    // Linear probing to find or claim a cell
    // We use cell_counts as both existence marker and count
    atomic_fetch_add_explicit(&cell_counts[h], 1, memory_order_relaxed);

    // Accumulate position using fixed-point arithmetic (multiply by 1000, store as uint)
    uint fx = uint(int(px * 1000.0) + 2000000);  // offset to make positive
    uint fy = uint(int(py * 1000.0) + 2000000);
    uint fz = uint(int(pz * 1000.0) + 2000000);

    atomic_fetch_add_explicit(&cell_sum_x[h], fx, memory_order_relaxed);
    atomic_fetch_add_explicit(&cell_sum_y[h], fy, memory_order_relaxed);
    atomic_fetch_add_explicit(&cell_sum_z[h], fz, memory_order_relaxed);
}

// Pass 2: average and compact occupied cells
kernel void voxel_average(
    device uint*                   cell_counts [[buffer(0)]],  // non-atomic read
    device uint*                   cell_sum_x  [[buffer(1)]],
    device uint*                   cell_sum_y  [[buffer(2)]],
    device uint*                   cell_sum_z  [[buffer(3)]],
    device PointXYZ*               output      [[buffer(4)]],
    device atomic_uint*            out_count   [[buffer(5)]],
    constant VoxelDownsampleParams& params     [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.table_size) return;

    uint count = cell_counts[tid];
    if (count == 0) return;

    float fc = float(count);

    // Reverse the fixed-point encoding
    float avg_x = (float(int(cell_sum_x[tid]) - int(count) * 2000000) / 1000.0) / fc;
    float avg_y = (float(int(cell_sum_y[tid]) - int(count) * 2000000) / 1000.0) / fc;
    float avg_z = (float(int(cell_sum_z[tid]) - int(count) * 2000000) / 1000.0) / fc;

    uint slot = atomic_fetch_add_explicit(out_count, 1, memory_order_relaxed);

    output[slot].x = avg_x;
    output[slot].y = avg_y;
    output[slot].z = avg_z;
    output[slot]._pad = 0.0;
}
