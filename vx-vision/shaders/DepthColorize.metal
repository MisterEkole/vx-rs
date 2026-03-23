#include <metal_stdlib>
using namespace metal;

// ================================================================
// depth_colorize — map depth to RGBA via colormap (Turbo/Jet/Inferno).
// Dispatch: 2D per-pixel.
// ================================================================

struct DepthColorizeParams {
    float min_depth;
    float max_depth;
    uint  colormap_id;  // 0 = Turbo, 1 = Jet, 2 = Inferno
    uint  width;
    uint  height;
};

// Turbo colormap (256 entries, approximated via polynomial)
static float3 turbo_colormap(float t) {
    // Simplified turbo approximation
    float r = clamp(1.0 - 2.0 * abs(t - 0.75), 0.0, 1.0);
    float g = clamp(1.0 - 2.0 * abs(t - 0.5), 0.0, 1.0);
    float b = clamp(1.0 - 2.0 * abs(t - 0.25), 0.0, 1.0);

    // Better turbo approximation using cubic polynomials
    float4 kR = float4(0.13572, 4.61539, -42.6603, 77.1689);
    float4 kG = float4(0.09140, 2.26419, -4.64610, 0.05765);
    float4 kB = float4(0.10667, 12.6489, -60.5820, 86.4677);

    float4 tv = float4(1.0, t, t * t, t * t * t);
    r = clamp(dot(kR, tv), 0.0, 1.0);
    g = clamp(dot(kG, tv), 0.0, 1.0);
    b = clamp(dot(kB, tv), 0.0, 1.0);

    // Turbo has a specific shape - use a more accurate piecewise approximation
    if (t < 0.25) {
        r = t * 2.8;
        g = t * 1.2;
        b = 0.3 + t * 2.8;
    } else if (t < 0.5) {
        float s = (t - 0.25) * 4.0;
        r = 0.2 + s * 0.6;
        g = 0.8 + s * 0.2;
        b = 1.0 - s * 0.6;
    } else if (t < 0.75) {
        float s = (t - 0.5) * 4.0;
        r = 0.8 + s * 0.2;
        g = 1.0 - s * 0.4;
        b = 0.4 - s * 0.3;
    } else {
        float s = (t - 0.75) * 4.0;
        r = 1.0 - s * 0.3;
        g = 0.6 - s * 0.5;
        b = 0.1 - s * 0.1;
    }

    return float3(clamp(r, 0.0, 1.0), clamp(g, 0.0, 1.0), clamp(b, 0.0, 1.0));
}

static float3 jet_colormap(float t) {
    float r = clamp(1.5 - abs(4.0 * t - 3.0), 0.0, 1.0);
    float g = clamp(1.5 - abs(4.0 * t - 2.0), 0.0, 1.0);
    float b = clamp(1.5 - abs(4.0 * t - 1.0), 0.0, 1.0);
    return float3(r, g, b);
}

static float3 inferno_colormap(float t) {
    // Simplified inferno approximation
    float r = clamp(t * 3.0 - 0.5, 0.0, 1.0);
    float g = clamp(t * 2.0 - 0.5, 0.0, 1.0);
    float b = clamp(sin(t * 3.14159) * 1.2, 0.0, 1.0);
    return float3(r, g, b);
}

kernel void depth_colorize(
    texture2d<float, access::read>   depth_in  [[texture(0)]],
    texture2d<float, access::write>  color_out [[texture(1)]],
    constant DepthColorizeParams&    params    [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;

    float d = depth_in.read(gid).r;

    if (d <= 0.0) {
        color_out.write(float4(0.0, 0.0, 0.0, 1.0), gid);
        return;
    }

    // Normalize depth to [0, 1]
    float t = clamp((d - params.min_depth) / (params.max_depth - params.min_depth), 0.0, 1.0);

    float3 color;
    switch (params.colormap_id) {
        case 1:  color = jet_colormap(t); break;
        case 2:  color = inferno_colormap(t); break;
        default: color = turbo_colormap(t); break;
    }

    color_out.write(float4(color, 1.0), gid);
}
