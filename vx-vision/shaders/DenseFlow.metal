#include <metal_stdlib>
using namespace metal;

// ================================================================
// Horn-Schunck dense optical flow.
//
// Iterative variational method: minimize E = ΣΣ (Ix·u + Iy·v + It)²
//   + α² (|∇u|² + |∇v|²)
//
// Pass 1 (flow_derivatives): Compute spatial/temporal image derivatives
//   Ix, Iy, It from two consecutive frames.
//
// Pass 2 (horn_schunck_iterate): One Jacobi iteration of the flow
//   update. Run multiple times for convergence.
//
// Output: two R32Float textures (flow_u, flow_v) — per-pixel motion.
// ================================================================

struct FlowParams {
    uint  width;
    uint  height;
    float alpha;     // smoothness weight (typical: 1.0–100.0)
};

// Pass 1: Compute image derivatives
kernel void flow_derivatives(
    texture2d<float, access::read>   frame0  [[texture(0)]],
    texture2d<float, access::read>   frame1  [[texture(1)]],
    texture2d<float, access::write>  Ix_out  [[texture(2)]],
    texture2d<float, access::write>  Iy_out  [[texture(3)]],
    texture2d<float, access::write>  It_out  [[texture(4)]],
    constant FlowParams&             params  [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;

    int w = int(params.width)  - 1;
    int h = int(params.height) - 1;
    int x = int(gid.x);
    int y = int(gid.y);

    // Averaged derivatives over 2x2x2 cube for robustness
    float I000 = frame0.read(uint2(clamp(x,   0, w), clamp(y,   0, h))).r;
    float I100 = frame0.read(uint2(clamp(x+1, 0, w), clamp(y,   0, h))).r;
    float I010 = frame0.read(uint2(clamp(x,   0, w), clamp(y+1, 0, h))).r;
    float I110 = frame0.read(uint2(clamp(x+1, 0, w), clamp(y+1, 0, h))).r;
    float I001 = frame1.read(uint2(clamp(x,   0, w), clamp(y,   0, h))).r;
    float I101 = frame1.read(uint2(clamp(x+1, 0, w), clamp(y,   0, h))).r;
    float I011 = frame1.read(uint2(clamp(x,   0, w), clamp(y+1, 0, h))).r;
    float I111 = frame1.read(uint2(clamp(x+1, 0, w), clamp(y+1, 0, h))).r;

    // Ix: average of x-derivatives
    float Ix = 0.25 * ((I100 - I000) + (I110 - I010) + (I101 - I001) + (I111 - I011));
    // Iy: average of y-derivatives
    float Iy = 0.25 * ((I010 - I000) + (I110 - I100) + (I011 - I001) + (I111 - I101));
    // It: average of t-derivatives
    float It = 0.25 * ((I001 - I000) + (I101 - I100) + (I011 - I010) + (I111 - I110));

    Ix_out.write(float4(Ix, 0.0, 0.0, 1.0), gid);
    Iy_out.write(float4(Iy, 0.0, 0.0, 1.0), gid);
    It_out.write(float4(It, 0.0, 0.0, 1.0), gid);
}

// Pass 2: One Jacobi iteration
kernel void horn_schunck_iterate(
    texture2d<float, access::read>   Ix_tex   [[texture(0)]],
    texture2d<float, access::read>   Iy_tex   [[texture(1)]],
    texture2d<float, access::read>   It_tex   [[texture(2)]],
    texture2d<float, access::read>   u_in     [[texture(3)]],
    texture2d<float, access::read>   v_in     [[texture(4)]],
    texture2d<float, access::write>  u_out    [[texture(5)]],
    texture2d<float, access::write>  v_out    [[texture(6)]],
    constant FlowParams&             params   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;

    float Ix = Ix_tex.read(gid).r;
    float Iy = Iy_tex.read(gid).r;
    float It = It_tex.read(gid).r;

    int w = int(params.width) - 1;
    int h = int(params.height) - 1;
    int x = int(gid.x);
    int y = int(gid.y);

    // Laplacian approximation: average of 4-connected neighbors
    float u_avg = 0.25 * (
        u_in.read(uint2(clamp(x-1, 0, w), y)).r +
        u_in.read(uint2(clamp(x+1, 0, w), y)).r +
        u_in.read(uint2(x, clamp(y-1, 0, h))).r +
        u_in.read(uint2(x, clamp(y+1, 0, h))).r
    );

    float v_avg = 0.25 * (
        v_in.read(uint2(clamp(x-1, 0, w), y)).r +
        v_in.read(uint2(clamp(x+1, 0, w), y)).r +
        v_in.read(uint2(x, clamp(y-1, 0, h))).r +
        v_in.read(uint2(x, clamp(y+1, 0, h))).r
    );

    float alpha2 = params.alpha * params.alpha;
    float denom  = alpha2 + Ix * Ix + Iy * Iy;
    float P      = (Ix * u_avg + Iy * v_avg + It) / denom;

    float u_new = u_avg - Ix * P;
    float v_new = v_avg - Iy * P;

    u_out.write(float4(u_new, 0.0, 0.0, 1.0), gid);
    v_out.write(float4(v_new, 0.0, 0.0, 1.0), gid);
}
