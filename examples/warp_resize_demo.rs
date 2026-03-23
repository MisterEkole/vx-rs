// examples/warp_resize_demo.rs
//
// Demonstrates GPU image transforms:
//   - Bilinear resize (downscale + upscale)
//   - Affine warp (rotation)
//   - Perspective warp (homography)
//   - GPU pyramid building
//
// Run with:
//   cargo run --release --example warp_resize_demo -- path/to/image.png

use std::time::Instant;

use vx_vision::kernels::pyramid::PyramidBuilder;
use vx_vision::kernels::resize::ImageResize;
use vx_vision::kernels::warp::ImageWarp;
use vx_vision::Context;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("Usage: warp_resize_demo <image_path>");

    let img = image::open(&path).expect("Failed to open image").to_luma8();
    let (w, h) = img.dimensions();
    println!("Input: {}x{}\n", w, h);

    // ── GPU setup ──
    let ctx = Context::new().expect("No Metal GPU available");
    let resizer = ImageResize::new(&ctx).expect("Resize pipeline");
    let warper = ImageWarp::new(&ctx).expect("Warp pipeline");
    let pyr_builder = PyramidBuilder::new(&ctx).expect("Pyramid pipeline");

    let texture = ctx
        .texture_gray8(img.as_raw(), w, h)
        .expect("Failed to create texture");

    // ── 1. Resize: downscale to 50% ──
    let half_w = w / 2;
    let half_h = h / 2;
    let t1 = Instant::now();
    let small = ctx.texture_output_gray8(half_w, half_h).expect("output");
    resizer
        .apply(&ctx, &texture, &small)
        .expect("Resize down failed");
    let down_ms = t1.elapsed().as_secs_f64() * 1000.0;
    println!(
        "Resize {}x{} → {}x{}:  {:.2} ms",
        w, h, half_w, half_h, down_ms
    );

    // ── 2. Resize: upscale back to original ──
    let t2 = Instant::now();
    let big = ctx.texture_output_gray8(w, h).expect("output");
    resizer.apply(&ctx, &small, &big).expect("Resize up failed");
    let up_ms = t2.elapsed().as_secs_f64() * 1000.0;
    println!(
        "Resize {}x{} → {}x{}:  {:.2} ms",
        half_w, half_h, w, h, up_ms
    );

    // Verify round-trip quality (compare original vs down+up)
    let orig_pixels = texture.read_gray8();
    let roundtrip_pixels = big.read_gray8();
    let mse: f64 = orig_pixels
        .iter()
        .zip(roundtrip_pixels.iter())
        .map(|(&a, &b)| {
            let d = a as f64 - b as f64;
            d * d
        })
        .sum::<f64>()
        / orig_pixels.len() as f64;
    let psnr = if mse > 0.0 {
        10.0 * (255.0f64 * 255.0 / mse).log10()
    } else {
        f64::INFINITY
    };
    println!("  Round-trip PSNR: {:.1} dB (MSE={:.2})\n", psnr, mse);

    // ── 3. Affine warp: 15° rotation around center ──
    let angle: f32 = 15.0f32.to_radians();
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let cos_a = angle.cos();
    let sin_a = angle.sin();

    // Forward: rotate around center
    //   dst = R * (src - center) + center
    // Inverse (what the shader needs):
    //   src = R^-1 * (dst - center) + center
    let inv_matrix: [f32; 6] = [
        cos_a,
        sin_a,
        cx - cos_a * cx - sin_a * cy,
        -sin_a,
        cos_a,
        cy + sin_a * cx - cos_a * cy,
    ];

    let t3 = Instant::now();
    let rotated = ctx.texture_output_gray8(w, h).expect("output");
    warper
        .affine(&ctx, &texture, &rotated, &inv_matrix)
        .expect("Affine warp failed");
    let aff_ms = t3.elapsed().as_secs_f64() * 1000.0;
    println!("Affine warp (15° rotation):   {:.2} ms", aff_ms);

    // Verify: non-black pixels (inside the rotated region)
    let rotated_pixels = rotated.read_gray8();
    let nonblack = rotated_pixels.iter().filter(|&&p| p > 0).count();
    let fill_pct = nonblack as f64 / rotated_pixels.len() as f64 * 100.0;
    println!("  Fill coverage: {:.1}%\n", fill_pct);

    // ── 4. Perspective warp: mild keystone correction ──
    // Identity with slight perspective (simulate a tilted view)
    let inv_homography: [f32; 9] = [
        1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0002, 0.0001, 1.0, // subtle perspective distortion
    ];

    let t4 = Instant::now();
    let warped = ctx.texture_output_gray8(w, h).expect("output");
    warper
        .perspective(&ctx, &texture, &warped, &inv_homography)
        .expect("Perspective warp failed");
    let persp_ms = t4.elapsed().as_secs_f64() * 1000.0;
    println!("Perspective warp:             {:.2} ms", persp_ms);

    // ── 5. GPU Pyramid (4 levels) ──
    let t5 = Instant::now();
    let levels = pyr_builder
        .build(&ctx, &texture, 4)
        .expect("Pyramid build failed");
    let pyr_ms = t5.elapsed().as_secs_f64() * 1000.0;

    println!("\nGPU Pyramid (4 levels):        {:.2} ms", pyr_ms);
    println!("  Level 0: {}x{} (input)", w, h);
    for (i, level) in levels.iter().enumerate() {
        println!("  Level {}: {}x{}", i + 1, level.width(), level.height());
    }

    // ── Summary ──
    println!(
        "\nTotal GPU time: {:.2} ms",
        down_ms + up_ms + aff_ms + persp_ms + pyr_ms
    );
}
