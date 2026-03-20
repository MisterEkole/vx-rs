#include <metal_stdlib>
using namespace metal;

// ================================================================
// Template matching — normalized cross-correlation (NCC).
//
// Slides a template across the input image and computes NCC at
// each position. Output is a correlation map where higher values
// indicate better matches.
//
// NCC = Σ((I-μI)(T-μT)) / sqrt(Σ(I-μI)² · Σ(T-μT)²)
//
// The template mean and norm are precomputed on CPU (constant).
// Dispatch as 2D grid over valid output positions.
// ================================================================

struct TemplateParams {
    uint  img_width;
    uint  img_height;
    uint  tpl_width;
    uint  tpl_height;
    float tpl_mean;     // precomputed template mean
    float tpl_norm;     // precomputed sqrt(Σ(T - μT)²)
};

kernel void template_match_ncc(
    texture2d<float, access::read>   image     [[texture(0)]],
    texture2d<float, access::read>   tpl       [[texture(1)]],
    texture2d<float, access::write>  result    [[texture(2)]],
    constant TemplateParams&         params    [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint out_w = params.img_width  - params.tpl_width  + 1;
    uint out_h = params.img_height - params.tpl_height + 1;
    if (gid.x >= out_w || gid.y >= out_h) return;

    // Compute local image patch mean
    float patch_sum = 0.0;
    uint tw = params.tpl_width;
    uint th = params.tpl_height;
    float n = float(tw * th);

    for (uint ty = 0; ty < th; ty++) {
        for (uint tx = 0; tx < tw; tx++) {
            patch_sum += image.read(uint2(gid.x + tx, gid.y + ty)).r;
        }
    }
    float patch_mean = patch_sum / n;

    // Compute NCC numerator and image patch norm
    float numer = 0.0;
    float img_var = 0.0;

    for (uint ty = 0; ty < th; ty++) {
        for (uint tx = 0; tx < tw; tx++) {
            float img_val = image.read(uint2(gid.x + tx, gid.y + ty)).r;
            float tpl_val = tpl.read(uint2(tx, ty)).r;

            float id = img_val - patch_mean;
            float td = tpl_val - params.tpl_mean;

            numer   += id * td;
            img_var += id * id;
        }
    }

    float denom = sqrt(img_var) * params.tpl_norm;
    float ncc = (denom > 1e-8) ? (numer / denom) : 0.0;

    result.write(float4(ncc, 0.0, 0.0, 1.0), gid);
}
