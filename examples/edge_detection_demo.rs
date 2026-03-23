// examples/edge_detection_demo.rs
//
// Full edge detection pipeline:
//   Histogram EQ -> Sobel gradients -> Canny edges -> Morphological cleanup
//
// Run with:
//   cargo run --release --example edge_detection_demo -- path/to/image.png

use std::time::Instant;

use vx_vision::kernels::canny::{CannyConfig, CannyDetector};
use vx_vision::kernels::histogram::Histogram;
use vx_vision::kernels::morphology::{MorphConfig, Morphology};
use vx_vision::kernels::sobel::SobelFilter;
use vx_vision::Context;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("Usage: edge_detection_demo <image_path>");

    let img = image::open(&path).expect("Failed to open image").to_luma8();
    let (w, h) = img.dimensions();
    println!("Image: {}x{} ({})", w, h, path);

    let t0 = Instant::now();
    let ctx = Context::new().expect("No Metal GPU available");
    let sobel = SobelFilter::new(&ctx).expect("Sobel pipeline");
    let canny = CannyDetector::new(&ctx).expect("Canny pipeline");
    let hist = Histogram::new(&ctx).expect("Histogram pipeline");
    let morph = Morphology::new(&ctx).expect("Morphology pipeline");
    println!("GPU setup: {:.2} ms\n", t0.elapsed().as_secs_f64() * 1000.0);

    let texture = ctx
        .texture_gray8(img.as_raw(), w, h)
        .expect("Failed to create texture");

    // Histogram analysis
    let t1 = Instant::now();
    let bins = hist.compute(&ctx, &texture).expect("Histogram failed");
    let hist_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let total_pixels = (w as u64) * (h as u64);
    let mean_intensity = bins
        .iter()
        .enumerate()
        .map(|(i, &c)| i as f64 * c as f64)
        .sum::<f64>()
        / total_pixels as f64;
    let nonzero_bins = bins.iter().filter(|&&c| c > 0).count();

    println!("Histogram:        {:.2} ms", hist_ms);
    println!("  Mean intensity: {:.1} / 255", mean_intensity);
    println!("  Active bins:    {} / 256", nonzero_bins);

    // Histogram equalization
    let t2 = Instant::now();
    let equalized = ctx.texture_output_gray8(w, h).expect("output texture");
    hist.equalize(&ctx, &texture, &equalized)
        .expect("Equalization failed");
    let eq_ms = t2.elapsed().as_secs_f64() * 1000.0;
    println!("Equalization:     {:.2} ms", eq_ms);

    // Sobel gradients
    let t3 = Instant::now();
    let sobel_result = sobel.compute(&ctx, &equalized).expect("Sobel failed");
    let sobel_ms = t3.elapsed().as_secs_f64() * 1000.0;

    let mag_data = sobel_result.magnitude.read_r32float();
    let max_mag = mag_data.iter().cloned().fold(0.0f32, f32::max);
    let avg_mag = mag_data.iter().sum::<f32>() / mag_data.len() as f32;

    println!("Sobel gradients:  {:.2} ms", sobel_ms);
    println!("  Max magnitude:  {:.4}", max_mag);
    println!("  Avg magnitude:  {:.4}", avg_mag);

    // Canny edge detection
    let t4 = Instant::now();
    let mut canny_config = CannyConfig::default();
    canny_config.low_threshold = 0.04;
    canny_config.high_threshold = 0.12;
    canny_config.blur_sigma = 1.4;
    canny_config.blur_radius = 4;
    let edges = canny
        .detect(&ctx, &texture, &canny_config)
        .expect("Canny failed");
    let canny_ms = t4.elapsed().as_secs_f64() * 1000.0;

    let edge_data = edges.read_r32float();
    let edge_count = edge_data.iter().filter(|&&v| v > 0.5).count();
    let edge_pct = edge_count as f64 / total_pixels as f64 * 100.0;

    println!("Canny edges:      {:.2} ms", canny_ms);
    println!("  Edge pixels:    {} ({:.1}%)", edge_count, edge_pct);

    // Morphological cleanup (close small gaps in edges)
    let t5 = Instant::now();
    let closed = ctx.texture_output_gray8(w, h).expect("output texture");
    morph
        .close(&ctx, &edges, &closed, &MorphConfig::new(1, 1))
        .expect("Morphological close failed");
    let morph_ms = t5.elapsed().as_secs_f64() * 1000.0;

    let closed_data = closed.read_gray8();
    let closed_edge_count = closed_data.iter().filter(|&&v| v > 128).count();

    println!("Morph close:      {:.2} ms", morph_ms);
    println!(
        "  Edge pixels:    {} (was {})",
        closed_edge_count, edge_count
    );

    // Summary
    let total = hist_ms + eq_ms + sobel_ms + canny_ms + morph_ms;
    println!("\nTotal pipeline:   {:.2} ms", total);
    println!("  Histogram + EQ: {:.2} ms", hist_ms + eq_ms);
    println!("  Sobel:          {:.2} ms", sobel_ms);
    println!("  Canny:          {:.2} ms", canny_ms);
    println!("  Morphology:     {:.2} ms", morph_ms);
}
