#include <metal_stdlib>
using namespace metal;

// ================================================================
// Color space conversions.
//
// rgba_to_gray:  RGBA → single-channel grayscale (ITU-R BT.601)
// gray_to_rgba:  Single-channel → RGBA (broadcast gray to RGB, A=1)
// rgba_to_hsv:   RGBA → HSV (H in [0,1], S in [0,1], V in [0,1])
// hsv_to_rgba:   HSV → RGBA
//
// All dispatched as 2D grids: one thread per pixel.
// ================================================================

struct ColorParams {
    uint width;
    uint height;
};

// RGBA (4-channel) → Grayscale (1-channel)
// Uses BT.601 luminance weights: 0.299R + 0.587G + 0.114B
kernel void rgba_to_gray(
    texture2d<float, access::read>   input  [[texture(0)]],
    texture2d<float, access::write>  output [[texture(1)]],
    constant ColorParams&            params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;

    float4 rgba = input.read(gid);
    float gray = 0.299 * rgba.r + 0.587 * rgba.g + 0.114 * rgba.b;
    output.write(float4(gray, 0.0, 0.0, 1.0), gid);
}

// Grayscale (1-channel) → RGBA (4-channel)
kernel void gray_to_rgba(
    texture2d<float, access::read>   input  [[texture(0)]],
    texture2d<float, access::write>  output [[texture(1)]],
    constant ColorParams&            params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;

    float gray = input.read(gid).r;
    output.write(float4(gray, gray, gray, 1.0), gid);
}

// RGBA → HSV
// H in [0, 1] (representing 0–360°), S in [0, 1], V in [0, 1]
// Stored as: texture.r = H, texture.g = S, texture.b = V, texture.a = 1
kernel void rgba_to_hsv(
    texture2d<float, access::read>   input  [[texture(0)]],
    texture2d<float, access::write>  output [[texture(1)]],
    constant ColorParams&            params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;

    float4 rgba = input.read(gid);
    float r = rgba.r, g = rgba.g, b = rgba.b;

    float cmax = max(r, max(g, b));
    float cmin = min(r, min(g, b));
    float delta = cmax - cmin;

    // Value
    float v = cmax;

    // Saturation
    float s = (cmax > 0.0) ? delta / cmax : 0.0;

    // Hue
    float h = 0.0;
    if (delta > 0.0) {
        if (cmax == r) {
            h = (g - b) / delta;
            if (h < 0.0) h += 6.0;
        } else if (cmax == g) {
            h = 2.0 + (b - r) / delta;
        } else {
            h = 4.0 + (r - g) / delta;
        }
        h /= 6.0; // normalize to [0, 1]
    }

    output.write(float4(h, s, v, 1.0), gid);
}

// HSV → RGBA
kernel void hsv_to_rgba(
    texture2d<float, access::read>   input  [[texture(0)]],
    texture2d<float, access::write>  output [[texture(1)]],
    constant ColorParams&            params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;

    float4 hsv = input.read(gid);
    float h = hsv.r * 6.0; // [0, 6)
    float s = hsv.g;
    float v = hsv.b;

    float c = v * s;
    float x = c * (1.0 - abs(fmod(h, 2.0) - 1.0));
    float m = v - c;

    float3 rgb;
    if      (h < 1.0) rgb = float3(c, x, 0);
    else if (h < 2.0) rgb = float3(x, c, 0);
    else if (h < 3.0) rgb = float3(0, c, x);
    else if (h < 4.0) rgb = float3(0, x, c);
    else if (h < 5.0) rgb = float3(x, 0, c);
    else              rgb = float3(c, 0, x);

    output.write(float4(rgb + m, 1.0), gid);
}
