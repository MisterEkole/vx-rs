// examples/feature_matching_demo.rs
//
// Feature matching using three approaches:
//   1. ORB (FAST + Harris + ORB descriptor + brute-force Hamming)
//   2. SIFT-like (DoG keypoints + 128-dim descriptors + L2)
//   3. Template matching (NCC)
//
// Run with:
//   cargo run --release --example feature_matching_demo -- image1.png image2.png

use std::time::Instant;

use vx_vision::kernels::fast::{FastDetectConfig, FastDetector};
use vx_vision::kernels::harris::{HarrisConfig, HarrisScorer};
use vx_vision::kernels::matcher::{BruteMatcher, MatchConfig};
use vx_vision::kernels::nms::{NmsConfig, NmsSuppressor};
use vx_vision::kernels::orb::{OrbConfig, OrbDescriptor};
use vx_vision::kernels::sift::{SiftConfig, SiftPipeline};
use vx_vision::kernels::template_match::TemplateMatcher;
use vx_vision::Context;

/// Standard ORB test pattern (256 pairs x 4 offsets = 1024 values).
fn orb_pattern() -> Vec<i32> {
    let mut pattern = Vec::with_capacity(1024);
    let mut rng: u32 = 0xDEADBEEF;
    for _ in 0..1024 {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        let val = ((rng % 31) as i32) - 15;
        pattern.push(val);
    }
    pattern
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: feature_matching_demo <image1> <image2>");
        std::process::exit(1);
    }

    let img1 = image::open(&args[1])
        .expect("Failed to open image1")
        .to_luma8();
    let img2 = image::open(&args[2])
        .expect("Failed to open image2")
        .to_luma8();
    let (w1, h1) = img1.dimensions();
    let (w2, h2) = img2.dimensions();
    println!("Image 1: {}x{} ({})", w1, h1, &args[1]);
    println!("Image 2: {}x{} ({})\n", w2, h2, &args[2]);

    let t0 = Instant::now();
    let ctx = Context::new().expect("No Metal GPU");
    let fast = FastDetector::new(&ctx).expect("FAST pipeline");
    let harris = HarrisScorer::new(&ctx).expect("Harris pipeline");
    let nms = NmsSuppressor::new(&ctx).expect("NMS pipeline");
    let orb = OrbDescriptor::new(&ctx).expect("ORB pipeline");
    let brute = BruteMatcher::new(&ctx).expect("Matcher pipeline");
    let sift = SiftPipeline::new(&ctx).expect("SIFT pipeline");
    let tm = TemplateMatcher::new(&ctx).expect("Template matching pipeline");
    println!("GPU setup: {:.2} ms\n", t0.elapsed().as_secs_f64() * 1000.0);

    let tex1 = ctx.texture_gray8(img1.as_raw(), w1, h1).expect("texture1");
    let tex2 = ctx.texture_gray8(img2.as_raw(), w2, h2).expect("texture2");

    // ORB pipeline
    println!("=== ORB Pipeline ===");

    let pattern = orb_pattern();
    let t_orb = Instant::now();

    let fast_cfg = FastDetectConfig::new(20, 4096);
    let harris_cfg = HarrisConfig::new(0.04, 3);
    let nms_cfg = NmsConfig::new(8.0);

    let corners1 = fast.detect(&ctx, &tex1, &fast_cfg).expect("FAST1").corners;
    let scored1 = harris
        .compute(&ctx, &tex1, &corners1, &harris_cfg)
        .expect("Harris1");
    let nms1 = nms.run(&ctx, &scored1, &nms_cfg).expect("NMS1");
    let kp1: Vec<_> = nms1.into_iter().take(512).collect();
    let desc1 = orb
        .compute(&ctx, &tex1, &kp1, &pattern, &OrbConfig::default())
        .expect("ORB1");

    let corners2 = fast.detect(&ctx, &tex2, &fast_cfg).expect("FAST2").corners;
    let scored2 = harris
        .compute(&ctx, &tex2, &corners2, &harris_cfg)
        .expect("Harris2");
    let nms2 = nms.run(&ctx, &scored2, &nms_cfg).expect("NMS2");
    let kp2: Vec<_> = nms2.into_iter().take(512).collect();
    let desc2 = orb
        .compute(&ctx, &tex2, &kp2, &pattern, &OrbConfig::default())
        .expect("ORB2");

    let orb_detect_ms = t_orb.elapsed().as_secs_f64() * 1000.0;

    let flat1: Vec<u32> = desc1.descriptors.iter().flat_map(|d| d.desc).collect();
    let flat2: Vec<u32> = desc2.descriptors.iter().flat_map(|d| d.desc).collect();

    let t_match = Instant::now();
    let mut match_cfg = MatchConfig::default();
    match_cfg.max_hamming = 64;
    match_cfg.ratio_thresh = 0.75;
    let orb_matches = brute
        .match_descriptors(&ctx, &flat1, &flat2, &match_cfg)
        .expect("Matching");
    let orb_match_ms = t_match.elapsed().as_secs_f64() * 1000.0;

    println!("  Keypoints:   {} / {}", kp1.len(), kp2.len());
    println!("  Detect+Desc: {:.2} ms", orb_detect_ms);
    println!("  Match:       {:.2} ms", orb_match_ms);
    println!("  Matches:     {}", orb_matches.len());
    if !orb_matches.is_empty() {
        let avg_dist: f32 =
            orb_matches.iter().map(|m| m.distance as f32).sum::<f32>() / orb_matches.len() as f32;
        let avg_ratio: f32 =
            orb_matches.iter().map(|m| m.ratio).sum::<f32>() / orb_matches.len() as f32;
        println!("  Avg Hamming: {:.1}", avg_dist);
        println!("  Avg ratio:   {:.3}", avg_ratio);

        for (i, m) in orb_matches.iter().take(5).enumerate() {
            let p1 = kp1[m.query_idx as usize].position;
            let p2 = kp2[m.train_idx as usize].position;
            println!(
                "    [{:2}] ({:.0},{:.0}) -> ({:.0},{:.0}) dist={} ratio={:.3}",
                i, p1[0], p1[1], p2[0], p2[1], m.distance, m.ratio
            );
        }
    }

    // SIFT-like pipeline
    println!("\n=== SIFT-like Pipeline ===");

    let mut sift_cfg = SiftConfig::default();
    sift_cfg.n_octaves = 3;
    sift_cfg.n_levels = 3;
    sift_cfg.contrast_threshold = 0.04;
    sift_cfg.max_keypoints = 1024;

    let t_sift = Instant::now();
    let feat1 = sift
        .detect_and_describe(&ctx, &tex1, &sift_cfg)
        .expect("SIFT1");
    let feat2 = sift
        .detect_and_describe(&ctx, &tex2, &sift_cfg)
        .expect("SIFT2");
    let sift_detect_ms = t_sift.elapsed().as_secs_f64() * 1000.0;

    let t_sift_match = Instant::now();
    let sift_matches = SiftPipeline::match_features(&feat1, &feat2, 0.7);
    let sift_match_ms = t_sift_match.elapsed().as_secs_f64() * 1000.0;

    println!("  Features:    {} / {}", feat1.len(), feat2.len());
    println!("  Detect+Desc: {:.2} ms", sift_detect_ms);
    println!("  Match:       {:.2} ms", sift_match_ms);
    println!("  Matches:     {}", sift_matches.len());
    if !sift_matches.is_empty() {
        let avg_dist: f32 =
            sift_matches.iter().map(|m| m.distance).sum::<f32>() / sift_matches.len() as f32;
        println!("  Avg L2 dist: {:.4}", avg_dist);
        for (i, m) in sift_matches.iter().take(5).enumerate() {
            let q = &feat1[m.query_idx];
            let t = &feat2[m.train_idx];
            println!(
                "    [{:2}] ({:.0},{:.0}) -> ({:.0},{:.0}) dist={:.4} ratio={:.3}",
                i, q.x, q.y, t.x, t.y, m.distance, m.ratio
            );
        }
    }

    // Template matching (extract patch from img1, find in img2)
    println!("\n=== Template Matching ===");

    let patch_size = 64u32;
    if w1 >= patch_size && h1 >= patch_size && w2 >= patch_size && h2 >= patch_size {
        let cx = (w1 / 2 - patch_size / 2) as usize;
        let cy = (h1 / 2 - patch_size / 2) as usize;
        let mut patch = vec![0u8; (patch_size * patch_size) as usize];
        for y in 0..patch_size as usize {
            for x in 0..patch_size as usize {
                patch[y * patch_size as usize + x] =
                    img1.as_raw()[(cy + y) * w1 as usize + (cx + x)];
            }
        }
        let tpl = ctx
            .texture_gray8(&patch, patch_size, patch_size)
            .expect("template texture");

        let t_tm = Instant::now();
        let tm_result = tm
            .match_template(&ctx, &tex2, &tpl)
            .expect("Template match");
        let tm_ms = t_tm.elapsed().as_secs_f64() * 1000.0;

        println!("  Patch from:  ({}, {}) in image 1", cx, cy);
        println!(
            "  Best match:  ({}, {}) in image 2",
            tm_result.best_x, tm_result.best_y
        );
        println!("  NCC score:   {:.4}", tm_result.best_score);
        println!("  Time:        {:.2} ms", tm_ms);
    } else {
        println!(
            "  (Images too small for {}x{} template)",
            patch_size, patch_size
        );
    }

    // Summary
    println!("\n--- Summary ---");
    println!(
        "ORB:   {} matches in {:.2} ms (detect+match)",
        orb_matches.len(),
        orb_detect_ms + orb_match_ms
    );
    println!(
        "SIFT:  {} matches in {:.2} ms (detect+match)",
        sift_matches.len(),
        sift_detect_ms + sift_match_ms
    );
}
