// examples/threshold_demo.rs
//
// Demonstrates thresholding and segmentation:
//   Otsu, adaptive threshold (GPU integral image), and fixed binary.
//
// Run with:
//   cargo run --release --example threshold_demo -- path/to/image.png

use std::time::Instant;

use vx_vision::kernels::histogram::Histogram;
use vx_vision::kernels::integral::IntegralImage;
use vx_vision::kernels::threshold::{AdaptiveThresholdConfig, Threshold};
use vx_vision::Context;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("Usage: threshold_demo <image_path>");

    let img = image::open(&path).expect("Failed to open image").to_luma8();
    let (w, h) = img.dimensions();
    println!("Image: {}x{}\n", w, h);

    let ctx = Context::new().expect("No Metal GPU available");
    let thresh = Threshold::new(&ctx).expect("Threshold pipeline");
    let integral = IntegralImage::new(&ctx).expect("Integral pipeline");
    let hist = Histogram::new(&ctx).expect("Histogram pipeline");

    let texture = ctx
        .texture_gray8(img.as_raw(), w, h)
        .expect("Failed to create texture");

    // Histogram
    let t1 = Instant::now();
    let bins = hist.compute(&ctx, &texture).expect("Histogram failed");
    let hist_ms = t1.elapsed().as_secs_f64() * 1000.0;

    println!("Histogram:          {:.2} ms", hist_ms);
    print!("  Distribution:     ");
    for chunk in bins.chunks(32) {
        let sum: u32 = chunk.iter().sum();
        let bar_len = (sum as f64 / (w as f64 * h as f64) * 40.0) as usize;
        print!("{}", "#".repeat(bar_len.clamp(0, 5)));
    }
    println!();

    // Otsu's method
    let t2 = Instant::now();
    let otsu_output = ctx.texture_output_gray8(w, h).expect("output");
    let otsu_val = thresh
        .otsu(&ctx, &texture, &otsu_output)
        .expect("Otsu failed");
    let otsu_ms = t2.elapsed().as_secs_f64() * 1000.0;

    let otsu_pixels = otsu_output.read_gray8();
    let white_count = otsu_pixels.iter().filter(|&&v| v > 128).count();
    let white_pct = white_count as f64 / otsu_pixels.len() as f64 * 100.0;

    println!("\nOtsu threshold:     {:.2} ms", otsu_ms);
    println!(
        "  Threshold:        {:.3} ({}/255)",
        otsu_val,
        (otsu_val * 255.0) as u32
    );
    println!("  Foreground:       {:.1}% white", white_pct);

    // Fixed binary threshold at 0.5
    let t3 = Instant::now();
    let binary_output = ctx.texture_output_gray8(w, h).expect("output");
    thresh
        .binary(&ctx, &texture, &binary_output, 0.5, false)
        .expect("Binary threshold failed");
    let binary_ms = t3.elapsed().as_secs_f64() * 1000.0;

    let binary_pixels = binary_output.read_gray8();
    let binary_white = binary_pixels.iter().filter(|&&v| v > 128).count();

    println!("\nBinary (t=0.5):     {:.2} ms", binary_ms);
    println!(
        "  Foreground:       {:.1}% white",
        binary_white as f64 / binary_pixels.len() as f64 * 100.0
    );

    // Integral image
    let t4 = Instant::now();
    let integral_tex = integral
        .compute(&ctx, &texture)
        .expect("Integral image failed");
    let integral_ms = t4.elapsed().as_secs_f64() * 1000.0;

    let integral_data = integral_tex.read_r32float();
    let total_sum = integral_data.last().unwrap_or(&0.0);
    let expected_sum: f64 = img.as_raw().iter().map(|&v| v as f64 / 255.0).sum();

    println!("\nIntegral image:     {:.2} ms", integral_ms);
    println!(
        "  Total sum:        {:.1} (expected ~{:.1})",
        total_sum, expected_sum
    );

    // Adaptive threshold
    let t5 = Instant::now();
    let adaptive_output = ctx.texture_output_gray8(w, h).expect("output");
    let adaptive_config = AdaptiveThresholdConfig::new(15, 0.03, false);
    thresh
        .adaptive(
            &ctx,
            &texture,
            &integral_tex,
            &adaptive_output,
            &adaptive_config,
        )
        .expect("Adaptive threshold failed");
    let adaptive_ms = t5.elapsed().as_secs_f64() * 1000.0;

    let adaptive_pixels = adaptive_output.read_gray8();
    let adaptive_white = adaptive_pixels.iter().filter(|&&v| v > 128).count();

    println!("\nAdaptive (r=15):    {:.2} ms", adaptive_ms);
    println!(
        "  Foreground:       {:.1}% white",
        adaptive_white as f64 / adaptive_pixels.len() as f64 * 100.0
    );

    // Summary
    println!("\n{}", "=".repeat(50));
    println!(
        "Total pipeline:     {:.2} ms",
        hist_ms + otsu_ms + binary_ms + integral_ms + adaptive_ms
    );
    println!("  Histogram:        {:.2} ms", hist_ms);
    println!("  Otsu:             {:.2} ms", otsu_ms);
    println!("  Binary:           {:.2} ms", binary_ms);
    println!("  Integral image:   {:.2} ms", integral_ms);
    println!("  Adaptive:         {:.2} ms", adaptive_ms);
}
