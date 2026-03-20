#include <metal_stdlib>
using namespace metal;

// ================================================================
// Difference of Gaussians (DoG) — scale-space keypoint detection.
//
// dog_subtract:  Compute DoG = blur(sigma_k+1) - blur(sigma_k)
//   for adjacent Gaussian-blurred images. One thread per pixel.
//
// dog_extrema:   Find scale-space extrema (local min/max) in a
//   3×3×3 neighborhood across three adjacent DoG levels.
//   A pixel is an extremum if it is strictly greater (or less) than
//   all 26 neighbors (8 in current level + 9 above + 9 below).
//   Output: keypoint positions via atomic append.
//
// Full pipeline: build Gaussian pyramid → compute DoG between
//   adjacent levels → find extrema across 3 consecutive DoGs.
// ================================================================

struct DoGParams {
    uint width;
    uint height;
};

// Subtract two blurred images to produce a DoG level
kernel void dog_subtract(
    texture2d<float, access::read>   blur_a  [[texture(0)]],  // lower sigma
    texture2d<float, access::read>   blur_b  [[texture(1)]],  // higher sigma
    texture2d<float, access::write>  dog_out [[texture(2)]],
    constant DoGParams&              params  [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;

    float a = blur_a.read(gid).r;
    float b = blur_b.read(gid).r;
    dog_out.write(float4(b - a, 0.0, 0.0, 1.0), gid);
}

struct DoGExtremaParams {
    uint  width;
    uint  height;
    float contrast_threshold;  // minimum |DoG| to accept (reject low-contrast)
    uint  max_keypoints;
    uint  octave;              // current octave index (for output)
    uint  level;               // current level within octave (for output)
};

struct DoGKeypoint {
    float2 position;
    float  response;   // DoG value at extremum
    uint   octave;
    uint   level;
    float  _pad0;
    float  _pad1;
    float  _pad2;
};

// Find scale-space extrema across 3 DoG levels (below, current, above)
kernel void dog_extrema(
    texture2d<float, access::read>  dog_below   [[texture(0)]],
    texture2d<float, access::read>  dog_current [[texture(1)]],
    texture2d<float, access::read>  dog_above   [[texture(2)]],
    device DoGKeypoint*             keypoints   [[buffer(0)]],
    device atomic_uint*             kp_count    [[buffer(1)]],
    constant DoGExtremaParams&      params      [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    // Skip border pixels (need 3x3 neighborhood)
    if (gid.x < 1 || gid.x >= params.width - 1 ||
        gid.y < 1 || gid.y >= params.height - 1) return;

    float val = dog_current.read(gid).r;

    // Contrast threshold
    if (abs(val) < params.contrast_threshold) return;

    // Check if local maximum or minimum across 26 neighbors
    bool is_max = true;
    bool is_min = true;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            uint2 npos = uint2(int2(gid) + int2(dx, dy));

            // Check all 3 levels
            float below = dog_below.read(npos).r;
            float above = dog_above.read(npos).r;

            if (below >= val) is_max = false;
            if (below <= val) is_min = false;
            if (above >= val) is_max = false;
            if (above <= val) is_min = false;

            // Current level (skip center pixel)
            if (dx != 0 || dy != 0) {
                float curr = dog_current.read(npos).r;
                if (curr >= val) is_max = false;
                if (curr <= val) is_min = false;
            }

            if (!is_max && !is_min) return;
        }
    }

    if (!is_max && !is_min) return;

    // Append keypoint
    uint slot = atomic_fetch_add_explicit(kp_count, 1, memory_order_relaxed);
    if (slot >= params.max_keypoints) return;

    keypoints[slot].position = float2(gid);
    keypoints[slot].response = val;
    keypoints[slot].octave = params.octave;
    keypoints[slot].level = params.level;
}
