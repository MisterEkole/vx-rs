// examples/orb_stereo_benchmark.rs
//
// Benchmark: FAST -> Harris -> NMS -> ORB describe -> StereoMatch
// on synchronised stereo image pairs (EuRoC MAV dataset format).
//
// Run with:
//   cargo run --release --example orb_stereo_benchmark -- \
//       /path/to/mav0/cam0/data/ /path/to/mav0/cam1/data/
//
// The --release flag matters -- debug builds bottleneck on image decoding.

use std::path::PathBuf;
use std::time::Instant;

use vx_vision::kernels::fast::{FastDetectConfig, FastDetector};
use vx_vision::kernels::harris::{HarrisConfig, HarrisScorer};
use vx_vision::kernels::nms::{NmsConfig, NmsSuppressor};
use vx_vision::kernels::orb::{OrbConfig, OrbDescriptor};
use vx_vision::kernels::stereomatch::{StereoConfig, StereoMatcher};
use vx_vision::Context;

const FAST_THRESHOLD: i32 = 20;
const MAX_CORNERS: u32 = 4096;
const HARRIS_THRESHOLD: f32 = 1e-5;
const MAX_KEYPOINTS: usize = 512;
const NMS_RADIUS: f32 = 8.0;

// EuRoC cam0 approximate intrinsics (Machine Hall sequence)
const FX: f32 = 458.654;
const FY: f32 = 457.296;
const CX: f32 = 367.215;
const CY: f32 = 248.375;
const BASELINE: f32 = 0.110; // metres

fn main() {
    let mut args = std::env::args().skip(1);
    let left_dir = args
        .next()
        .expect("Usage: orb_stereo_benchmark <left_dir> <right_dir>");
    let right_dir = args
        .next()
        .expect("Usage: orb_stereo_benchmark <left_dir> <right_dir>");

    let left_frames = collect_frames(&left_dir);

    // Match by filename (timestamp) -- keep only frames present in both dirs
    let pairs: Vec<(PathBuf, PathBuf)> = left_frames
        .into_iter()
        .filter_map(|l| {
            let name = l.file_name()?;
            let r = PathBuf::from(&right_dir).join(name);
            if r.exists() {
                Some((l, r))
            } else {
                None
            }
        })
        .collect();

    if pairs.is_empty() {
        eprintln!("No matching stereo pairs found between the two directories.");
        return;
    }

    println!("Found {} stereo pairs", pairs.len());
    println!("Left dir:  {}", left_dir);
    println!("Right dir: {}", right_dir);
    println!("{}\n", "-".repeat(74));

    let t0 = Instant::now();
    let ctx = Context::new().expect("No Metal GPU available");
    let fast = FastDetector::new(&ctx).expect("FAST pipeline");
    let harris = HarrisScorer::new(&ctx).expect("Harris pipeline");
    let nms = NmsSuppressor::new(&ctx).expect("NMS pipeline");
    let orb = OrbDescriptor::new(&ctx).expect("ORB pipeline");
    let matcher = StereoMatcher::new(&ctx).expect("StereoMatcher pipeline");
    let setup_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("GPU setup: {:.2} ms  (context + 5 pipelines)\n", setup_ms);

    let fast_cfg = FastDetectConfig::new(FAST_THRESHOLD, MAX_CORNERS);
    let harris_cfg = HarrisConfig::new(0.04, 3);
    let nms_cfg = NmsConfig::new(NMS_RADIUS);
    let orb_cfg = OrbConfig::new(15);
    let mut stereo_cfg = StereoConfig::default();
    stereo_cfg.max_epipolar = 2.0;
    stereo_cfg.min_disparity = 1.0;
    stereo_cfg.max_disparity = 120.0;
    stereo_cfg.max_hamming = 50;
    stereo_cfg.ratio_thresh = 0.8;
    stereo_cfg.fx = FX;
    stereo_cfg.fy = FY;
    stereo_cfg.cx = CX;
    stereo_cfg.cy = CY;
    stereo_cfg.baseline = BASELINE;

    // Same seed every run for reproducible results
    let pattern = orb_pattern();

    let mut acc_load = Acc::default();
    let mut acc_fast = Acc::default();
    let mut acc_harris = Acc::default();
    let mut acc_nms = Acc::default();
    let mut acc_orb = Acc::default();
    let mut acc_match = Acc::default();
    let mut acc_total = Acc::default();

    let mut total_matches: u64 = 0;
    let mut total_depth_sum: f64 = 0.0;
    let mut total_depth_n: u64 = 0;

    println!(
        "{:>5}  {:>5} {:>5}  {:>5}  {:>7}  {:>6} {:>6} {:>6} {:>5} {:>6} {:>7}  {:>7}",
        "Frame",
        "LKpts",
        "RKpts",
        "Match",
        "AvgZ(m)",
        "Load",
        "FAST",
        "Harris",
        "NMS",
        "ORB",
        "Stereo",
        "Total"
    );
    println!("{}", "-".repeat(90));

    let benchmark_start = Instant::now();

    for (frame_idx, (left_path, right_path)) in pairs.iter().enumerate() {
        let t_frame = Instant::now();

        let t = Instant::now();
        let (left_px, lw, lh) = load_gray8(left_path);
        let (right_px, rw, rh) = load_gray8(right_path);
        let load_ms = t.elapsed().as_secs_f64() * 1000.0;
        acc_load.push(load_ms);

        let left_tex = ctx.texture_gray8(&left_px, lw, lh).expect("left texture");
        let right_tex = ctx.texture_gray8(&right_px, rw, rh).expect("right texture");

        let t = Instant::now();
        let left_fast = fast.detect(&ctx, &left_tex, &fast_cfg).expect("FAST left");
        let right_fast = fast
            .detect(&ctx, &right_tex, &fast_cfg)
            .expect("FAST right");
        let fast_ms = t.elapsed().as_secs_f64() * 1000.0;
        acc_fast.push(fast_ms);

        if left_fast.corners.is_empty() || right_fast.corners.is_empty() {
            continue;
        }

        let t = Instant::now();
        let mut left_corners = harris
            .compute(&ctx, &left_tex, &left_fast.corners, &harris_cfg)
            .expect("Harris left");
        let mut right_corners = harris
            .compute(&ctx, &right_tex, &right_fast.corners, &harris_cfg)
            .expect("Harris right");
        let harris_ms = t.elapsed().as_secs_f64() * 1000.0;
        acc_harris.push(harris_ms);

        left_corners.retain(|c| c.response > HARRIS_THRESHOLD);
        right_corners.retain(|c| c.response > HARRIS_THRESHOLD);
        left_corners.sort_by(|a, b| b.response.partial_cmp(&a.response).unwrap());
        right_corners.sort_by(|a, b| b.response.partial_cmp(&a.response).unwrap());
        left_corners.truncate(MAX_KEYPOINTS * 4);
        right_corners.truncate(MAX_KEYPOINTS * 4);

        if left_corners.is_empty() || right_corners.is_empty() {
            continue;
        }

        let t = Instant::now();
        let mut left_kpts = nms.run(&ctx, &left_corners, &nms_cfg).expect("NMS left");
        let mut right_kpts = nms.run(&ctx, &right_corners, &nms_cfg).expect("NMS right");
        let nms_ms = t.elapsed().as_secs_f64() * 1000.0;
        acc_nms.push(nms_ms);

        left_kpts.sort_by(|a, b| b.response.partial_cmp(&a.response).unwrap());
        right_kpts.sort_by(|a, b| b.response.partial_cmp(&a.response).unwrap());
        left_kpts.truncate(MAX_KEYPOINTS);
        right_kpts.truncate(MAX_KEYPOINTS);

        if left_kpts.is_empty() || right_kpts.is_empty() {
            continue;
        }

        let n_left = left_kpts.len();
        let n_right = right_kpts.len();

        let t = Instant::now();
        let left_orb = orb
            .compute(&ctx, &left_tex, &left_kpts, &pattern, &orb_cfg)
            .expect("ORB left");
        let right_orb = orb
            .compute(&ctx, &right_tex, &right_kpts, &pattern, &orb_cfg)
            .expect("ORB right");
        let orb_ms = t.elapsed().as_secs_f64() * 1000.0;
        acc_orb.push(orb_ms);

        let t = Instant::now();
        let stereo = matcher
            .run(
                &ctx,
                &left_kpts,
                &right_kpts,
                &left_orb.descriptors,
                &right_orb.descriptors,
                &stereo_cfg,
            )
            .expect("StereoMatch failed");
        let match_ms = t.elapsed().as_secs_f64() * 1000.0;
        acc_match.push(match_ms);

        let n_matches = stereo.matches.len();
        total_matches += n_matches as u64;

        let (depth_sum, depth_n) = stereo
            .matches
            .iter()
            .filter(|m| m.point_3d[2] > 0.0 && m.point_3d[2] < 200.0)
            .fold((0.0f64, 0u32), |(s, n), m| {
                (s + m.point_3d[2] as f64, n + 1)
            });
        let avg_z_m = if depth_n > 0 {
            depth_sum / depth_n as f64
        } else {
            0.0
        };
        total_depth_sum += depth_sum;
        total_depth_n += depth_n as u64;

        let frame_ms = t_frame.elapsed().as_secs_f64() * 1000.0;
        acc_total.push(frame_ms);

        let i = frame_idx + 1;
        if i <= 5 || i % 25 == 0 || i == pairs.len() {
            println!(
                "{:>5}  {:>5} {:>5}  {:>5}  {:>7.2}  {:>6.2} {:>6.2} {:>6.2} {:>5.2} {:>6.2} {:>7.2}  {:>7.2}",
                i, n_left, n_right, n_matches,
                avg_z_m,
                load_ms, fast_ms, harris_ms, nms_ms, orb_ms, match_ms, frame_ms
            );
        }
    }

    let wall_s = benchmark_start.elapsed().as_secs_f64();

    // Summary
    println!("\n{}", "=".repeat(70));
    println!("BENCHMARK SUMMARY -- ORB + StereoMatch");
    println!("{}", "=".repeat(70));
    println!("Stereo pairs processed:    {}", acc_total.n);
    println!("Wall-clock time:           {:.2} s", wall_s);
    println!(
        "Average throughput:        {:.1} pairs/s",
        acc_total.n as f64 / wall_s
    );
    println!("Average pair latency:      {:.2} ms", acc_total.mean());
    println!();

    println!("Per-stage averages (both images where applicable):");
    println!("  Image load (x2):         {:>7.2} ms", acc_load.mean());
    println!("  FAST detect (x2):        {:>7.2} ms", acc_fast.mean());
    println!("  Harris score (x2):       {:>7.2} ms", acc_harris.mean());
    println!("  NMS (x2):                {:>7.2} ms", acc_nms.mean());
    println!("  ORB describe (x2):       {:>7.2} ms", acc_orb.mean());
    println!("  StereoMatch:             {:>7.2} ms", acc_match.mean());
    println!();

    println!("Per-stage maximums:");
    println!("  Image load (x2):         {:>7.2} ms", acc_load.max);
    println!("  FAST detect (x2):        {:>7.2} ms", acc_fast.max);
    println!("  Harris score (x2):       {:>7.2} ms", acc_harris.max);
    println!("  NMS (x2):                {:>7.2} ms", acc_nms.max);
    println!("  ORB describe (x2):       {:>7.2} ms", acc_orb.max);
    println!("  StereoMatch:             {:>7.2} ms", acc_match.max);
    println!();

    println!("Match statistics:");
    println!("  Total matches:           {}", total_matches);
    println!(
        "  Avg matches/pair:        {:.1}",
        total_matches as f64 / acc_total.n as f64
    );
    if total_depth_n > 0 {
        println!(
            "  Avg triangulated depth:  {:.2} m",
            total_depth_sum / total_depth_n as f64
        );
    }
    println!();

    println!("Pipeline config:");
    println!("  FAST threshold:          {}", FAST_THRESHOLD);
    println!(
        "  Harris k / filter:       0.04 / response > {}",
        HARRIS_THRESHOLD
    );
    println!("  NMS radius:              {} px", NMS_RADIUS);
    println!("  Max keypoints/image:     {}", MAX_KEYPOINTS);
    println!("  ORB patch radius:        {}", orb_cfg.patch_radius);
    println!("  Stereo max Hamming:      {}", stereo_cfg.max_hamming);
    println!(
        "  Stereo max disparity:    {:.0} px",
        stereo_cfg.max_disparity
    );
    println!(
        "  Stereo epipolar tol:     {:.1} px",
        stereo_cfg.max_epipolar
    );
    println!("  Camera fx/fy:            {:.1} / {:.1}", FX, FY);
    println!("  Baseline:                {:.3} m", BASELINE);
}

fn collect_frames(dir: &str) -> Vec<PathBuf> {
    let mut frames: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("Cannot read directory {dir}: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "png"))
        .collect();
    frames.sort();
    frames
}

fn load_gray8(path: &PathBuf) -> (Vec<u8>, u32, u32) {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("Failed to load {}: {e}", path.display()))
        .to_luma8();
    let (w, h) = img.dimensions();
    (img.into_raw(), w, h)
}

/// Generate a fixed, reproducible ORB test-pair pattern.
/// 256 pairs of (dx1, dy1, dx2, dy2), each in [-12, 12].
fn orb_pattern() -> Vec<i32> {
    let mut state: u64 = 0x4F52_425F_5041_5454; // "ORB_PATT"
    let mut next = || -> i32 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (((state >> 33) as i32).abs() % 25) - 12
    };
    (0..1024).map(|_| next()).collect()
}

#[derive(Default)]
struct Acc {
    sum: f64,
    max: f64,
    n: u64,
}

impl Acc {
    fn push(&mut self, v: f64) {
        self.sum += v;
        if v > self.max {
            self.max = v;
        }
        self.n += 1;
    }
    fn mean(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.sum / self.n as f64
        }
    }
}
