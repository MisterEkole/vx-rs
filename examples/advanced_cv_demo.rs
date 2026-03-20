// examples/advanced_cv_demo.rs
//
// Demonstrates the advanced CV algorithms in vx-vision:
//   - Bilateral filter (edge-preserving smoothing)
//   - Canny edge detection → Hough line detection
//   - Distance transform (JFA)
//   - Connected components labeling
//   - Template matching (self-patch NCC)
//
// Run with:
//   cargo run --release --example advanced_cv_demo -- path/to/image.png

use std::time::Instant;

use vx_vision::Context;
use vx_vision::kernels::bilateral::{BilateralConfig, BilateralFilter};
use vx_vision::kernels::canny::{CannyConfig, CannyDetector};
use vx_vision::kernels::connected::{CCLConfig, ConnectedComponents};
use vx_vision::kernels::distance::{DistanceConfig, DistanceTransform};
use vx_vision::kernels::hough::{HoughConfig, HoughLines};
use vx_vision::kernels::template_match::TemplateMatcher;
use vx_vision::kernels::threshold::Threshold;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("Usage: advanced_cv_demo <image_path>");

    // ── Load image ──
    let img = image::open(&path)
        .expect("Failed to open image")
        .to_luma8();
    let (w, h) = img.dimensions();
    println!("Image: {}x{} ({})\n", w, h, path);

    // ── GPU setup ──
    let t0 = Instant::now();
    let ctx = Context::new().expect("No Metal GPU available");
    let bilateral = BilateralFilter::new(&ctx).expect("Bilateral pipeline");
    let canny = CannyDetector::new(&ctx).expect("Canny pipeline");
    let hough = HoughLines::new(&ctx).expect("Hough pipeline");
    let dt = DistanceTransform::new(&ctx).expect("Distance transform pipeline");
    let ccl = ConnectedComponents::new(&ctx).expect("CCL pipeline");
    let tm = TemplateMatcher::new(&ctx).expect("Template matching pipeline");
    let thresh = Threshold::new(&ctx).expect("Threshold pipeline");
    println!("GPU setup: {:.2} ms\n", t0.elapsed().as_secs_f64() * 1000.0);

    let texture = ctx.texture_gray8(img.as_raw(), w, h)
        .expect("Failed to create texture");

    // ── 1. Bilateral filter ──
    let t1 = Instant::now();
    let filtered = ctx.texture_output_gray8(w, h).expect("output");
    bilateral.apply(&ctx, &texture, &filtered, &BilateralConfig {
        radius: 5,
        sigma_spatial: 10.0,
        sigma_range: 0.1,
    }).expect("Bilateral failed");
    let bilateral_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let orig_data = texture.read_gray8();
    let filt_data = filtered.read_gray8();
    let mse: f64 = orig_data.iter().zip(filt_data.iter())
        .map(|(&a, &b)| { let d = a as f64 - b as f64; d * d })
        .sum::<f64>() / orig_data.len() as f64;
    let psnr = if mse > 0.0 { 10.0 * (255.0f64 * 255.0 / mse).log10() } else { f64::INFINITY };

    println!("1. Bilateral filter:   {:.2} ms", bilateral_ms);
    println!("   PSNR vs original:   {:.1} dB (lower = more smoothing)", psnr);

    // ── 2. Canny → Hough lines ──
    let t2 = Instant::now();
    let edges = canny.detect(&ctx, &texture, &CannyConfig {
        low_threshold: 0.04,
        high_threshold: 0.12,
        blur_sigma: 1.4,
        blur_radius: 4,
    }).expect("Canny failed");
    let canny_ms = t2.elapsed().as_secs_f64() * 1000.0;

    let edge_data = edges.read_r32float();
    let edge_count = edge_data.iter().filter(|&&v| v > 0.5).count();

    println!("\n2. Canny edges:        {:.2} ms", canny_ms);
    println!("   Edge pixels:        {}", edge_count);

    let t2b = Instant::now();
    let lines = hough.detect(&ctx, &edges, &HoughConfig {
        n_theta: 180,
        edge_threshold: 0.5,
        vote_threshold: (w.min(h) / 4) as u32,
        max_lines: 64,
        nms_radius: 5,
    }).expect("Hough failed");
    let hough_ms = t2b.elapsed().as_secs_f64() * 1000.0;

    println!("   Hough lines:        {:.2} ms", hough_ms);
    println!("   Lines detected:     {}", lines.len());
    for (i, line) in lines.iter().take(5).enumerate() {
        println!("     [{:2}] rho={:7.1} theta={:5.1}° votes={}",
            i, line.rho, line.theta.to_degrees(), line.votes);
    }
    if lines.len() > 5 {
        println!("     ... and {} more", lines.len() - 5);
    }

    // ── 3. Otsu threshold → Distance transform ──
    let t3 = Instant::now();
    let binary = ctx.texture_output_gray8(w, h).expect("output");
    let otsu_val = thresh.otsu(&ctx, &texture, &binary).expect("Otsu failed");
    let otsu_ms = t3.elapsed().as_secs_f64() * 1000.0;

    println!("\n3. Otsu threshold:     {:.2} ms (threshold = {})", otsu_ms, otsu_val);

    let t3b = Instant::now();
    let dist_map = dt.compute(&ctx, &binary, &DistanceConfig { threshold: 0.5 })
        .expect("Distance transform failed");
    let dt_ms = t3b.elapsed().as_secs_f64() * 1000.0;

    let dist_data = dist_map.read_r32float();
    let max_dist = dist_data.iter().cloned().fold(0.0f32, f32::max);
    let avg_dist = dist_data.iter().sum::<f32>() / dist_data.len() as f32;

    println!("   Distance transform: {:.2} ms", dt_ms);
    println!("   Max distance:       {:.1} px", max_dist);
    println!("   Avg distance:       {:.1} px", avg_dist);

    // ── 4. Connected components ──
    let t4 = Instant::now();
    let ccl_result = ccl.label(&ctx, &binary, &CCLConfig {
        threshold: 0.5,
        max_iterations: 256,
    }).expect("CCL failed");
    let ccl_ms = t4.elapsed().as_secs_f64() * 1000.0;

    println!("\n4. Connected components: {:.2} ms", ccl_ms);
    println!("   Components:          {}", ccl_result.n_components);
    println!("   Iterations:          {}", ccl_result.iterations);

    // ── 5. Template matching (self-patch) ──
    // Extract a 32x32 patch from center of image as template
    let patch_size = 32u32;
    if w >= patch_size && h >= patch_size {
        let cx = (w / 2 - patch_size / 2) as usize;
        let cy = (h / 2 - patch_size / 2) as usize;
        let mut patch = vec![0u8; (patch_size * patch_size) as usize];
        for y in 0..patch_size as usize {
            for x in 0..patch_size as usize {
                patch[y * patch_size as usize + x] = img.as_raw()[(cy + y) * w as usize + (cx + x)];
            }
        }
        let tpl_tex = ctx.texture_gray8(&patch, patch_size, patch_size)
            .expect("Template texture");

        let t5 = Instant::now();
        let match_result = tm.match_template(&ctx, &texture, &tpl_tex)
            .expect("Template match failed");
        let tm_ms = t5.elapsed().as_secs_f64() * 1000.0;

        println!("\n5. Template matching:   {:.2} ms ({}x{} patch)", tm_ms, patch_size, patch_size);
        println!("   Best match:          ({}, {}) score={:.4}", match_result.best_x, match_result.best_y, match_result.best_score);
        println!("   Expected:            ({}, {})", cx, cy);
        let dx = match_result.best_x as i32 - cx as i32;
        let dy = match_result.best_y as i32 - cy as i32;
        println!("   Error:               ({}, {}) px", dx, dy);
    }

    // ── Summary ──
    let total = bilateral_ms + canny_ms + hough_ms + otsu_ms + dt_ms + ccl_ms;
    println!("\n─── Summary ─────────────────────────");
    println!("Bilateral:       {:.2} ms", bilateral_ms);
    println!("Canny + Hough:   {:.2} ms", canny_ms + hough_ms);
    println!("Otsu + DT:       {:.2} ms", otsu_ms + dt_ms);
    println!("CCL:             {:.2} ms", ccl_ms);
    println!("Total:           {:.2} ms", total);
}
