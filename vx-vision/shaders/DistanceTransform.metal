#include <metal_stdlib>
using namespace metal;

// ================================================================
// Distance transform — Jump Flooding Algorithm (JFA).
//
// Computes the Euclidean distance from each pixel to the nearest
// "seed" (zero-valued) pixel.
//
// jfa_init:  Initialize seed map. Seed pixels store their own
//            coordinates; non-seed pixels store (0xFFFF, 0xFFFF).
//
// jfa_step:  One JFA step at a given step size k. Each pixel checks
//            8 neighbors at offset ±k and updates to the nearest
//            seed found. Run for k = N/2, N/4, ..., 2, 1
//            where N = max(width, height) rounded up to power of 2.
//
// jfa_distance: Convert seed map to distance values (R32Float).
//
// All dispatched as 2D grids: one thread per pixel.
// ================================================================

struct JFAParams {
    uint  width;
    uint  height;
    int   step_size;
    float threshold;   // pixels <= threshold are seeds
};

// Initialize: seeds store own coords, others store sentinel
kernel void jfa_init(
    texture2d<float, access::read>   input  [[texture(0)]],
    device uint2*                    seeds  [[buffer(0)]],
    constant JFAParams&              params [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;

    uint idx = gid.y * params.width + gid.x;
    float val = input.read(gid).r;

    if (val <= params.threshold) {
        seeds[idx] = gid;  // this pixel is a seed
    } else {
        seeds[idx] = uint2(0xFFFF, 0xFFFF);  // sentinel
    }
}

// One JFA step: check 8 neighbors at offset ±step_size
kernel void jfa_step(
    device uint2*        seeds_in  [[buffer(0)]],
    device uint2*        seeds_out [[buffer(1)]],
    constant JFAParams&  params    [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;

    uint idx = gid.y * params.width + gid.x;
    uint2 best_seed = seeds_in[idx];
    float best_dist2 = 1e18;

    if (best_seed.x != 0xFFFF) {
        float dx = float(gid.x) - float(best_seed.x);
        float dy = float(gid.y) - float(best_seed.y);
        best_dist2 = dx * dx + dy * dy;
    }

    int k = params.step_size;
    int w = int(params.width);
    int h = int(params.height);

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;

            int nx = int(gid.x) + dx * k;
            int ny = int(gid.y) + dy * k;

            if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;

            uint nidx = uint(ny) * params.width + uint(nx);
            uint2 neighbor_seed = seeds_in[nidx];

            if (neighbor_seed.x == 0xFFFF) continue;

            float sdx = float(gid.x) - float(neighbor_seed.x);
            float sdy = float(gid.y) - float(neighbor_seed.y);
            float dist2 = sdx * sdx + sdy * sdy;

            if (dist2 < best_dist2) {
                best_dist2 = dist2;
                best_seed = neighbor_seed;
            }
        }
    }

    seeds_out[idx] = best_seed;
}

// Convert seed map to Euclidean distance
kernel void jfa_distance(
    device uint2*                    seeds  [[buffer(0)]],
    texture2d<float, access::write>  output [[texture(0)]],
    constant JFAParams&              params [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;

    uint idx = gid.y * params.width + gid.x;
    uint2 seed = seeds[idx];

    float dist = 0.0;
    if (seed.x != 0xFFFF) {
        float dx = float(gid.x) - float(seed.x);
        float dy = float(gid.y) - float(seed.y);
        dist = sqrt(dx * dx + dy * dy);
    }

    output.write(float4(dist, 0.0, 0.0, 1.0), gid);
}
