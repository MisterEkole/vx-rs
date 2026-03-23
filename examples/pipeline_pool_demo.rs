// examples/pipeline_pool_demo.rs
//
// Simulates a real-time camera pipeline:
//   Frame -> Bilateral denoise -> Sobel edges -> Canny -> Hough lines
//
// Shows the performance benefit of Pipeline (batched dispatches) and
// TexturePool (recycled texture allocations).
//
// Run with:
//   cargo run --release --example pipeline_pool_demo -- path/to/image.png

use std::time::Instant;

use vx_vision::{Context, TexturePool};
use vx_vision::kernels::bilateral::{BilateralConfig, BilateralFilter};
use vx_vision::kernels::canny::{CannyConfig, CannyDetector};
use vx_vision::kernels::histogram::Histogram;
use vx_vision::kernels::sobel::SobelFilter;
use vx_vision::kernels::threshold::Threshold;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("Usage: pipeline_pool_demo <image_path>");

    let img = image::open(&path).expect("Failed to open image").to_luma8();
    let (w, h) = img.dimensions();
    println!("Image: {}x{} ({})\n", w, h, path);

    let ctx = Context::new().expect("No Metal GPU");
    let bilateral = BilateralFilter::new(&ctx).expect("Bilateral");
    let canny = CannyDetector::new(&ctx).expect("Canny");
    let sobel = SobelFilter::new(&ctx).expect("Sobel");
    let hist = Histogram::new(&ctx).expect("Histogram");
    let thresh = Threshold::new(&ctx).expect("Threshold");

    let texture = ctx.texture_gray8(img.as_raw(), w, h).expect("input texture");

    // Individual command buffers (no pooling, no pipeline)
    println!("=== Without Pipeline/Pool (individual dispatch) ===");
    let n_frames = 10;
    let t_individual = Instant::now();

    for _ in 0..n_frames {
        let filtered = ctx.texture_output_gray8(w, h).expect("output");
        bilateral.apply(&ctx, &texture, &filtered, &BilateralConfig::new(3, 5.0, 0.1))
            .expect("bilateral");

        let _sobel_result = sobel.compute(&ctx, &filtered).expect("sobel");

        let mut canny_cfg = CannyConfig::default();
        canny_cfg.low_threshold = 0.04;
        canny_cfg.high_threshold = 0.12;
        canny_cfg.blur_sigma = 1.4;
        canny_cfg.blur_radius = 4;
        let _edges = canny.detect(&ctx, &texture, &canny_cfg).expect("canny");

        let binary = ctx.texture_output_gray8(w, h).expect("output");
        let _ = thresh.otsu(&ctx, &texture, &binary).expect("otsu");

        let _ = hist.compute(&ctx, &texture).expect("histogram");
    }

    let individual_ms = t_individual.elapsed().as_secs_f64() * 1000.0;
    let per_frame_individual = individual_ms / n_frames as f64;
    println!("  {} frames: {:.2} ms total", n_frames, individual_ms);
    println!("  Per frame: {:.2} ms", per_frame_individual);
    println!("  FPS:       {:.1}\n", 1000.0 / per_frame_individual);

    // With TexturePool (recycled allocations)
    println!("=== With TexturePool (recycled allocations) ===");
    let mut pool = TexturePool::new();

    let t_pooled = Instant::now();

    for _ in 0..n_frames {
        let filtered = pool.acquire_gray8(&ctx, w, h).expect("pooled output");
        bilateral.apply(&ctx, &texture, &filtered, &BilateralConfig::new(3, 5.0, 0.1))
            .expect("bilateral");

        let sobel_result = sobel.compute(&ctx, &filtered).expect("sobel");

        let mut canny_cfg2 = CannyConfig::default();
        canny_cfg2.low_threshold = 0.04;
        canny_cfg2.high_threshold = 0.12;
        canny_cfg2.blur_sigma = 1.4;
        canny_cfg2.blur_radius = 4;
        let edges = canny.detect(&ctx, &texture, &canny_cfg2).expect("canny");

        let binary = pool.acquire_gray8(&ctx, w, h).expect("pooled output");
        let _ = thresh.otsu(&ctx, &texture, &binary).expect("otsu");

        let _ = hist.compute(&ctx, &texture).expect("histogram");

        pool.release(filtered);
        pool.release(sobel_result.grad_x);
        pool.release(sobel_result.grad_y);
        pool.release(sobel_result.magnitude);
        pool.release(sobel_result.direction);
        pool.release(edges);
        pool.release(binary);
    }

    let pooled_ms = t_pooled.elapsed().as_secs_f64() * 1000.0;
    let per_frame_pooled = pooled_ms / n_frames as f64;
    println!("  {} frames: {:.2} ms total", n_frames, pooled_ms);
    println!("  Per frame: {:.2} ms", per_frame_pooled);
    println!("  FPS:       {:.1}", 1000.0 / per_frame_pooled);
    println!("  Pool stats: {} acquires, {} hits ({:.0}% reuse)",
        pool.total_acquires(), pool.total_hits(),
        pool.hit_rate() * 100.0);
    println!("  Cached:    {} textures", pool.cached_count());

    let speedup = per_frame_individual / per_frame_pooled;
    println!("  Speedup:   {:.2}x vs individual\n", speedup);

    // Pipeline API demo (encode-only, single command buffer)
    println!("=== Pipeline API Demo ===");
    println!("  The Pipeline builder lets you batch multiple kernel");
    println!("  dispatches into one command buffer. Example usage:\n");
    println!("    let mut pipe = Pipeline::begin(&ctx)?;");
    println!("    blur.encode(&ctx, pipe.cmd_buf(), &input, &blurred, &cfg)?;");
    println!("    // ... more encodes ...");
    println!("    let retained = pipe.commit_and_wait();\n");

    let blur = vx_vision::kernels::gaussian::GaussianBlur::new(&ctx).expect("Gaussian");
    let mut blur_cfg = vx_vision::kernels::gaussian::GaussianConfig::default();
    blur_cfg.sigma = 1.5;
    blur_cfg.radius = 4;

    let t_pipe = Instant::now();

    for _ in 0..n_frames {
        let blurred = pool.acquire_r32float(&ctx, w, h).expect("pooled");
        let output = pool.acquire_gray8(&ctx, w, h).expect("pooled");

        let pipe = vx_vision::Pipeline::begin(&ctx).expect("pipeline");
        let _state = blur.encode(&ctx, pipe.cmd_buf(), &texture, &blurred, &blur_cfg)
            .expect("encode blur");

        let retained = pipe.commit_and_wait();
        pool.release_all(retained.into_iter());

        pool.release(blurred);
        pool.release(output);
    }

    let pipe_ms = t_pipe.elapsed().as_secs_f64() * 1000.0;
    let per_frame_pipe = pipe_ms / n_frames as f64;
    println!("  Pipeline + Pool: {:.2} ms/frame", per_frame_pipe);
    println!("  Pool hit rate:   {:.0}%\n", pool.hit_rate() * 100.0);

    // Summary
    println!("--- Summary ---");
    println!("Individual:      {:.2} ms/frame", per_frame_individual);
    println!("With Pool:       {:.2} ms/frame ({:.2}x faster)",
        per_frame_pooled, per_frame_individual / per_frame_pooled);
    println!("Final pool:      {} cached, {:.0}% hit rate",
        pool.cached_count(), pool.hit_rate() * 100.0);
}
