// examples/klt_benchmark.rs
//
// End-to-end benchmark: FAST detection → Harris scoring → KLT tracking
// across sequential frames from the EuRoC MAV dataset.
//
// EuRoC dataset structure:
//   mav0/cam0/data/
//     1403636579763555584.png
//     1403636579813555456.png
//     ...
//
// Run with:
//   cargo run --release --example klt_benchmark -- /path/to/mav0/cam0/data/
//
// The --release flag matters — debug builds bottleneck on image decoding.

use std::path::PathBuf;
use std::time::Instant;

use vx_vision::Context;
use vx_vision::Texture;
use vx_vision::kernels::fast::{FastDetectConfig, FastDetector};
use vx_vision::kernels::harris::{HarrisConfig, HarrisScorer};
use vx_vision::kernels::klt::{ImagePyramid, KltConfig, KltTracker};
use vx_vision::kernels::pyramid::PyramidBuilder;

/// Minimum Harris response to keep a corner for tracking.
const HARRIS_THRESHOLD: f32 = 1e-5;

/// Maximum number of corners to track.
const MAX_TRACK_POINTS: usize = 500;

/// Re-detect corners when tracked count drops below this fraction.
const REDETECT_RATIO: f32 = 0.5;

fn main() {
    let data_dir = std::env::args()
        .nth(1)
        .expect("Usage: klt_benchmark <path/to/mav0/cam0/data/>");

    // ── Load and sort frames ──
    let mut frames: Vec<PathBuf> = std::fs::read_dir(&data_dir)
        .expect("Failed to read data directory")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |ext| ext == "png"))
        .collect();
    frames.sort();

    if frames.len() < 2 {
        eprintln!("Need at least 2 frames, found {}", frames.len());
        return;
    }

    println!("Found {} frames in {}", frames.len(), data_dir);
    println!("─────────────────────────────────────────────────────────────────\n");

    // ── GPU setup (once) ──
    let t0 = Instant::now();
    let ctx = Context::new().expect("No Metal GPU available");
    let fast = FastDetector::new(&ctx).expect("Failed to create FAST pipeline");
    let harris = HarrisScorer::new(&ctx).expect("Failed to create Harris pipeline");
    let tracker = KltTracker::new(&ctx).expect("Failed to create KLT pipeline");
    let pyr_builder = PyramidBuilder::new(&ctx).expect("Failed to create pyramid pipeline");
    let setup_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("GPU setup:        {:.2} ms (context + 4 pipelines)\n", setup_ms);

    // ── Configs ──
    let fast_config = FastDetectConfig {
        threshold: 20,
        max_corners: 4096,
    };
    let harris_config = HarrisConfig {
        k: 0.04,
        patch_radius: 3,
    };
    let klt_config = KltConfig::default();

    // ── Timing accumulators ──
    let mut total_load_ms = 0.0;
    let mut total_pyramid_ms = 0.0;
    let mut total_detect_ms = 0.0;
    let mut total_track_ms = 0.0;
    let mut total_frames = 0u64;
    let mut total_tracked = 0u64;
    let mut detect_count = 0u64;

    // ── Load first frame ──
    let (first_pixels, w, h) = load_gray8(&frames[0]);

    let first_texture = ctx.texture_gray8(&first_pixels, w, h)
        .expect("Failed to create texture");

    let t_pyr = Instant::now();
    let mut prev_pyramid = build_pyramid_gpu(&ctx, &pyr_builder, &first_texture);
    let first_pyr_ms = t_pyr.elapsed().as_secs_f64() * 1000.0;
    let mut tracked_positions = detect_and_score(
        &ctx, &fast, &harris, &first_texture,
        &fast_config, &harris_config,
    );
    detect_count += 1;
    let initial_count = tracked_positions.len();

    println!("First pyramid:    {:.2} ms ({}x{} -> 4 levels)", first_pyr_ms, w, h);
    println!("Initial corners:  {} (top {} after Harris filter)\n",
        initial_count, initial_count.min(MAX_TRACK_POINTS));

    println!("{:>6} {:>8} {:>7} {:>9} {:>9} {:>9} {:>9}",
        "Frame", "Tracked", "Input", "Load", "Pyramid", "Track", "Total");
    println!("{}", "-".repeat(68));

    // ── Main tracking loop ──
    let benchmark_start = Instant::now();

    for (i, frame_path) in frames.iter().enumerate().skip(1) {
        // Load frame
        let t_load = Instant::now();
        let (pixels, fw, fh) = load_gray8(frame_path);
        let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;
        total_load_ms += load_ms;

        // Upload frame + build GPU pyramid
        let curr_tex = ctx.texture_gray8(&pixels, fw, fh)
            .expect("Failed to create texture");

        let t_pyr = Instant::now();
        let curr_pyramid = build_pyramid_gpu(&ctx, &pyr_builder, &curr_tex);
        let pyr_ms = t_pyr.elapsed().as_secs_f64() * 1000.0;
        total_pyramid_ms += pyr_ms;

        // Re-detect if we've lost too many points
        let mut detect_ms = 0.0;
        let min_points = (initial_count as f32 * REDETECT_RATIO) as usize;
        if tracked_positions.len() < min_points {
            let t_det = Instant::now();
            tracked_positions = detect_and_score(
                &ctx, &fast, &harris, &curr_tex,
                &fast_config, &harris_config,
            );
            detect_ms = t_det.elapsed().as_secs_f64() * 1000.0;
            total_detect_ms += detect_ms;
            detect_count += 1;
        }

        let n_input = tracked_positions.len();

        // Track: prev pyramid -> curr pyramid
        let t_track = Instant::now();
        let result = tracker.track(
            &ctx,
            &prev_pyramid,
            &curr_pyramid,
            &tracked_positions,
            &klt_config,
        ).expect("KLT tracking failed");
        let track_ms = t_track.elapsed().as_secs_f64() * 1000.0;
        total_track_ms += track_ms;

        // Filter: keep only successfully tracked points
       
        tracked_positions = result.points.iter()
            .zip(result.status.iter())
            .filter(|(_, &ok)| ok)
            .map(|(&pt, _)| pt)
            .collect();
        let n_tracked = tracked_positions.len();

        total_tracked += n_tracked as u64;
        total_frames += 1;

        let frame_total = load_ms + pyr_ms + detect_ms + track_ms;

        // Print first 5, every 50th, and last frame
        if i < 5 || i % 50 == 0 || i == frames.len() - 1 {
            let det_str = if detect_ms > 0.0 {
                format!("  [redetect {:.1}ms]", detect_ms)
            } else {
                String::new()
            };
            println!("{:>6} {:>4}/{:<3} {:>6} {:>8.2} {:>8.2} {:>8.2} {:>8.2}{}",
                i, n_tracked, n_input, "", load_ms, pyr_ms, track_ms, frame_total, det_str);
        }

        // Advance
        prev_pyramid = curr_pyramid;
    }

    let wall_time = benchmark_start.elapsed().as_secs_f64();

    // ── Summary ──
    println!("\n{}", "=".repeat(68));
    println!("BENCHMARK SUMMARY");
    println!("{}", "=".repeat(68));
    println!("Frames processed:     {}", total_frames);
    println!("Wall-clock time:      {:.2} s", wall_time);
    println!("Avg FPS:              {:.1}", total_frames as f64 / wall_time);
    println!("Avg frame time:       {:.2} ms", (wall_time * 1000.0) / total_frames as f64);
    println!();
    println!("Per-stage averages:");
    println!("  Image load:         {:.2} ms", total_load_ms / total_frames as f64);
    println!("  Pyramid build:      {:.2} ms", total_pyramid_ms / total_frames as f64);
    println!("  KLT track:          {:.2} ms", total_track_ms / total_frames as f64);
    if detect_count > 0 {
        println!("  Detection (FAST+H): {:.2} ms (ran {} times)",
            total_detect_ms / detect_count as f64, detect_count);
    }
    println!();
    println!("Tracking stats:");
    println!("  Avg tracked/frame:  {:.0}", total_tracked as f64 / total_frames as f64);
    println!("  Re-detections:      {} (every ~{} frames)",
        detect_count,
        if detect_count > 1 { total_frames / (detect_count - 1) } else { total_frames }
    );
    println!();
    println!("Pipeline config:");
    println!("  FAST threshold:     {}", fast_config.threshold);
    println!("  Harris k:           {}", harris_config.k);
    println!("  Harris filter:      response > {}", HARRIS_THRESHOLD);
    println!("  Max track points:   {}", MAX_TRACK_POINTS);
    println!("  KLT window:         {}x{} (radius={})",
        2 * klt_config.win_radius + 1,
        2 * klt_config.win_radius + 1,
        klt_config.win_radius);
    println!("  KLT pyramid levels: {}", klt_config.max_level + 1);
    println!("  KLT max iterations: {}", klt_config.max_iterations);
    println!("  KLT epsilon:        {}", klt_config.epsilon);
    println!("  KLT min eigenvalue: {}", klt_config.min_eigenvalue);
    println!("  Redetect below:     {}%", (REDETECT_RATIO * 100.0) as u32);
}

// =========================================================================
// Helpers
// =========================================================================

/// Load a grayscale image, return (pixels, width, height).
fn load_gray8(path: &PathBuf) -> (Vec<u8>, u32, u32) {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("Failed to load {}: {e}", path.display()))
        .to_luma8();
    let (w, h) = img.dimensions();
    (img.into_raw(), w, h)
}

/// Build a 4-level image pyramid on the GPU using Gaussian downsample.
///
/// Level 0 = original texture, levels 1–3 are half-size each.
/// Uses the PyramidBuilder kernel instead of CPU box-filter downsampling.
fn build_pyramid_gpu(ctx: &Context, pyr: &PyramidBuilder, level0: &Texture) -> ImagePyramid {
    let down_levels = pyr.build(ctx, level0, 4)
        .expect("GPU pyramid build failed");

    // PyramidBuilder::build returns levels 1..N; we need [level0, l1, l2, l3].
    // We must re-upload level0 since ImagePyramid owns its textures.
    let l0_pixels = level0.read_gray8();
    let l0 = ctx.texture_gray8(&l0_pixels, level0.width(), level0.height())
        .expect("Failed to re-create level 0");

    let mut levels = down_levels.into_iter();
    ImagePyramid {
        levels: [
            l0,
            levels.next().expect("Missing pyramid level 1"),
            levels.next().expect("Missing pyramid level 2"),
            levels.next().expect("Missing pyramid level 3"),
        ],
    }
}

/// Run FAST -> Harris -> filter -> sort -> take top N.
/// Returns positions as [f32; 2] ready for KLT.
fn detect_and_score(
    ctx: &Context,
    fast: &FastDetector,
    harris: &HarrisScorer,
    texture: &Texture,
    fast_config: &FastDetectConfig,
    harris_config: &HarrisConfig,
) -> Vec<[f32; 2]> {
    let fast_result = fast.detect(ctx, texture, fast_config)
        .expect("FAST detection failed");

    if fast_result.corners.is_empty() {
        return Vec::new();
    }

    let mut scored = harris
        .compute(ctx, texture, &fast_result.corners, harris_config)
        .expect("Harris scoring failed");

    scored.retain(|c| c.response > HARRIS_THRESHOLD);
    scored.sort_by(|a, b| b.response.partial_cmp(&a.response).unwrap());
    scored.truncate(MAX_TRACK_POINTS);

    // Extract [f32; 2] positions -- what KLT expects
    scored.iter().map(|c| c.position).collect()
}