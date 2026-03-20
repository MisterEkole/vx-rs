#include <metal_stdlib>
using namespace metal;

// ================================================================
// Hough line transform — detect lines in edge images.
//
// hough_vote: Each edge pixel votes for all (rho, theta) lines
//   passing through it. Accumulates into a 2D voting buffer.
//   rho = x·cos(θ) + y·sin(θ)
//   2D dispatch over edge image pixels.
//
// hough_peaks: Find local maxima in the accumulator that exceed
//   a vote threshold. 2D dispatch over accumulator.
//
// Accumulator dimensions:
//   theta: [0, π) discretized into n_theta bins
//   rho:   [-diag, +diag) discretized into n_rho bins
// ================================================================

struct HoughVoteParams {
    uint  width;
    uint  height;
    uint  n_theta;       // number of angle bins (e.g. 180)
    uint  n_rho;         // number of distance bins
    float rho_max;       // diagonal length = sqrt(w² + h²)
    float edge_threshold; // minimum pixel value to consider as edge
};

// Vote accumulation: each edge pixel increments all lines through it
kernel void hough_vote(
    texture2d<float, access::read>  edges       [[texture(0)]],
    device atomic_uint*             accumulator [[buffer(0)]],
    constant HoughVoteParams&       params      [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;

    float val = edges.read(gid).r;
    if (val < params.edge_threshold) return;

    float x = float(gid.x);
    float y = float(gid.y);

    for (uint ti = 0; ti < params.n_theta; ti++) {
        float theta = float(ti) * M_PI_F / float(params.n_theta);
        float rho = x * cos(theta) + y * sin(theta);

        // Map rho from [-rho_max, rho_max] to [0, n_rho)
        float rho_norm = (rho + params.rho_max) / (2.0 * params.rho_max);
        uint ri = min(uint(rho_norm * float(params.n_rho)), params.n_rho - 1);

        atomic_fetch_add_explicit(&accumulator[ti * params.n_rho + ri], 1,
                                  memory_order_relaxed);
    }
}

struct HoughPeakParams {
    uint n_theta;
    uint n_rho;
    uint vote_threshold;   // minimum votes to be a peak
    uint max_lines;
    float rho_max;
    uint  nms_radius;      // suppression radius in accumulator space
};

struct HoughLine {
    float rho;
    float theta;
    uint  votes;
    uint  _pad;
};

// Peak detection with NMS in accumulator space
kernel void hough_peaks(
    device uint*              accumulator [[buffer(0)]],
    device HoughLine*         lines       [[buffer(1)]],
    device atomic_uint*       line_count  [[buffer(2)]],
    constant HoughPeakParams& params      [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= params.n_theta || gid.y >= params.n_rho) return;

    uint votes = accumulator[gid.x * params.n_rho + gid.y];
    if (votes < params.vote_threshold) return;

    // NMS: check if this is a local maximum
    int r = int(params.nms_radius);
    for (int dy = -r; dy <= r; dy++) {
        for (int dx = -r; dx <= r; dx++) {
            if (dx == 0 && dy == 0) continue;
            int nx = int(gid.x) + dx;
            int ny = int(gid.y) + dy;
            if (nx < 0 || nx >= int(params.n_theta) ||
                ny < 0 || ny >= int(params.n_rho)) continue;

            uint neighbor = accumulator[nx * params.n_rho + ny];
            if (neighbor > votes) return;  // not a maximum
            // Tie-breaking: lower index wins
            if (neighbor == votes && (uint(nx) * params.n_rho + uint(ny)) < (gid.x * params.n_rho + gid.y)) return;
        }
    }

    uint slot = atomic_fetch_add_explicit(line_count, 1, memory_order_relaxed);
    if (slot >= params.max_lines) return;

    float theta = float(gid.x) * M_PI_F / float(params.n_theta);
    float rho = float(gid.y) / float(params.n_rho) * 2.0 * params.rho_max - params.rho_max;

    lines[slot].rho   = rho;
    lines[slot].theta = theta;
    lines[slot].votes = votes;
}
