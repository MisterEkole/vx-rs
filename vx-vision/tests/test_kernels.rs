// Integration tests for GPU compute kernels.
//
// These tests exercise the full GPU pipeline: create textures, dispatch
// Metal shaders, read back results. They require an Apple Silicon or
// Metal-compatible GPU to run.
//
// Run with: cargo test -p vx-vision

use vx_vision::Context;

// ── Helper ──────────────────────────────────────────────────────────

fn make_gradient_image(w: u32, h: u32) -> Vec<u8> {
    (0..(w * h)).map(|i| {
        let x = i % w;
        ((x as f32 / w as f32) * 255.0) as u8
    }).collect()
}

fn make_checkerboard(w: u32, h: u32, block: u32) -> Vec<u8> {
    (0..(w * h)).map(|i| {
        let x = (i % w) / block;
        let y = (i / w) / block;
        if (x + y) % 2 == 0 { 255u8 } else { 0u8 }
    }).collect()
}

fn make_white_image(w: u32, h: u32) -> Vec<u8> {
    vec![255u8; (w * h) as usize]
}

fn make_black_image(w: u32, h: u32) -> Vec<u8> {
    vec![0u8; (w * h) as usize]
}

// ── Gaussian Blur ───────────────────────────────────────────────────

#[test]
fn gaussian_blur_reduces_noise() {
    let ctx = Context::new().unwrap();
    let blur = vx_vision::kernels::gaussian::GaussianBlur::new(&ctx).unwrap();

    let w = 64u32;
    let h = 64u32;
    // Noisy image: alternating 0 and 255
    let noisy: Vec<u8> = (0..(w * h)).map(|i| if i % 2 == 0 { 255 } else { 0 }).collect();
    let input = ctx.texture_gray8(&noisy, w, h).unwrap();
    let output = ctx.texture_output_gray8(w, h).unwrap();

    let cfg = vx_vision::kernels::gaussian::GaussianConfig { sigma: 2.0, radius: 6 };
    blur.apply(&ctx, &input, &output, &cfg).unwrap();

    let result = output.read_gray8();
    // After blur, values should be closer to 128 (mean of 0 and 255)
    let mean: f64 = result.iter().map(|&v| v as f64).sum::<f64>() / result.len() as f64;
    assert!((mean - 127.5).abs() < 20.0, "Mean after blur: {}", mean);

    // Variance should be significantly reduced
    let variance: f64 = result.iter().map(|&v| {
        let d = v as f64 - mean;
        d * d
    }).sum::<f64>() / result.len() as f64;
    let orig_variance: f64 = noisy.iter().map(|&v| {
        let d = v as f64 - 127.5;
        d * d
    }).sum::<f64>() / noisy.len() as f64;
    assert!(variance < orig_variance * 0.5, "Blur didn't reduce variance enough: {} vs {}", variance, orig_variance);
}

// ── Sobel ───────────────────────────────────────────────────────────

#[test]
fn sobel_detects_horizontal_gradient() {
    let ctx = Context::new().unwrap();
    let sobel = vx_vision::kernels::sobel::SobelFilter::new(&ctx).unwrap();

    let w = 64u32;
    let h = 64u32;
    let pixels = make_gradient_image(w, h);
    let input = ctx.texture_gray8(&pixels, w, h).unwrap();

    let result = sobel.compute(&ctx, &input).unwrap();

    let gx = result.grad_x.read_r32float();
    let gy = result.grad_y.read_r32float();
    let mag = result.magnitude.read_r32float();

    // Horizontal gradient should be detected in grad_x
    let avg_gx: f32 = gx.iter().sum::<f32>() / gx.len() as f32;
    let avg_gy: f32 = gy.iter().map(|v| v.abs()).sum::<f32>() / gy.len() as f32;

    assert!(avg_gx.abs() > avg_gy, "Should detect horizontal gradient: gx={} gy={}", avg_gx, avg_gy);

    // Magnitude should be > 0 for non-uniform image
    let max_mag = mag.iter().cloned().fold(0.0f32, f32::max);
    assert!(max_mag > 0.0, "Magnitude should be positive");
}

// ── FAST Corner Detection ───────────────────────────────────────────

#[test]
fn fast_detects_corners_in_checkerboard() {
    let ctx = Context::new().unwrap();
    let fast = vx_vision::kernels::fast::FastDetector::new(&ctx).unwrap();

    let w = 64u32;
    let h = 64u32;
    // Create a black image with a single bright rectangle. The edges
    // of the rectangle create corners that FAST-9 can detect (the
    // Bresenham circle of radius 3 straddles the bright/dark boundary).
    let mut pixels = vec![0u8; (w * h) as usize];
    for y in 16..48u32 {
        for x in 16..48u32 {
            pixels[(y * w + x) as usize] = 255;
        }
    }
    let input = ctx.texture_gray8(&pixels, w, h).unwrap();

    let cfg = vx_vision::kernels::fast::FastDetectConfig {
        threshold: 20,
        max_corners: 4096,
    };
    let result = fast.detect(&ctx, &input, &cfg).unwrap();

    // Rectangle edges should produce FAST corners
    assert!(!result.corners.is_empty(), "FAST should detect corners on rectangle edges (got 0)");
}

#[test]
fn fast_no_corners_in_uniform() {
    let ctx = Context::new().unwrap();
    let fast = vx_vision::kernels::fast::FastDetector::new(&ctx).unwrap();

    let w = 64u32;
    let h = 64u32;
    let pixels = make_white_image(w, h);
    let input = ctx.texture_gray8(&pixels, w, h).unwrap();

    let cfg = vx_vision::kernels::fast::FastDetectConfig {
        threshold: 20,
        max_corners: 1024,
    };
    let result = fast.detect(&ctx, &input, &cfg).unwrap();
    assert_eq!(result.corners.len(), 0, "Uniform image should have no corners");
}

// ── Histogram ───────────────────────────────────────────────────────

#[test]
fn histogram_uniform_white() {
    let ctx = Context::new().unwrap();
    let hist = vx_vision::kernels::histogram::Histogram::new(&ctx).unwrap();

    let w = 32u32;
    let h = 32u32;
    let pixels = make_white_image(w, h);
    let input = ctx.texture_gray8(&pixels, w, h).unwrap();

    let bins = hist.compute(&ctx, &input).unwrap();
    assert_eq!(bins[255], w * h, "All pixels should be in bin 255");
    assert_eq!(bins[0], 0, "No pixels should be in bin 0");
}

#[test]
fn histogram_uniform_black() {
    let ctx = Context::new().unwrap();
    let hist = vx_vision::kernels::histogram::Histogram::new(&ctx).unwrap();

    let w = 32u32;
    let h = 32u32;
    let pixels = make_black_image(w, h);
    let input = ctx.texture_gray8(&pixels, w, h).unwrap();

    let bins = hist.compute(&ctx, &input).unwrap();
    assert_eq!(bins[0], w * h, "All pixels should be in bin 0");
    let total: u32 = bins.iter().sum();
    assert_eq!(total, w * h, "Total should equal pixel count");
}

#[test]
fn histogram_equalization_increases_range() {
    let ctx = Context::new().unwrap();
    let hist = vx_vision::kernels::histogram::Histogram::new(&ctx).unwrap();

    let w = 64u32;
    let h = 64u32;
    // Low contrast image: values 100-150
    let pixels: Vec<u8> = (0..(w * h)).map(|i| 100 + (i % 51) as u8).collect();
    let input = ctx.texture_gray8(&pixels, w, h).unwrap();
    let output = ctx.texture_output_gray8(w, h).unwrap();

    hist.equalize(&ctx, &input, &output).unwrap();

    let result = output.read_gray8();
    let min_val = *result.iter().min().unwrap();
    let max_val = *result.iter().max().unwrap();
    let range = max_val as i32 - min_val as i32;

    // Equalized range should be much wider than original (50)
    assert!(range > 100, "Equalization should expand range: got {}", range);
}

// ── Threshold ───────────────────────────────────────────────────────

#[test]
fn threshold_binary() {
    let ctx = Context::new().unwrap();
    let thresh = vx_vision::kernels::threshold::Threshold::new(&ctx).unwrap();

    let w = 32u32;
    let h = 32u32;
    let pixels = make_gradient_image(w, h);
    let input = ctx.texture_gray8(&pixels, w, h).unwrap();
    let output = ctx.texture_output_gray8(w, h).unwrap();

    thresh.binary(&ctx, &input, &output, 0.5, false).unwrap();

    let result = output.read_gray8();
    for &val in &result {
        assert!(val == 0 || val == 255, "Binary should produce only 0 or 255, got {}", val);
    }
}

#[test]
fn threshold_otsu() {
    let ctx = Context::new().unwrap();
    let thresh = vx_vision::kernels::threshold::Threshold::new(&ctx).unwrap();

    let w = 64u32;
    let h = 64u32;
    // Bimodal image: half dark, half bright
    let mut pixels = vec![0u8; (w * h) as usize];
    for i in 0..(w * h / 2) as usize {
        pixels[i] = 50;
    }
    for i in (w * h / 2) as usize..(w * h) as usize {
        pixels[i] = 200;
    }
    let input = ctx.texture_gray8(&pixels, w, h).unwrap();
    let output = ctx.texture_output_gray8(w, h).unwrap();

    let otsu_val = thresh.otsu(&ctx, &input, &output).unwrap();

    // Otsu threshold is normalized [0,1]; should be between the two modes (50/255 ≈ 0.196, 200/255 ≈ 0.784)
    assert!(otsu_val > 0.15 && otsu_val < 0.85,
        "Otsu threshold should be between modes: got {}", otsu_val);

    let result = output.read_gray8();
    for &val in &result {
        assert!(val == 0 || val == 255, "Otsu output should be binary");
    }
}

// ── Morphology ──────────────────────────────────────────────────────

#[test]
fn morphology_erode_shrinks_white() {
    let ctx = Context::new().unwrap();
    let morph = vx_vision::kernels::morphology::Morphology::new(&ctx).unwrap();

    let w = 32u32;
    let h = 32u32;
    // White square in center
    let mut pixels = vec![0u8; (w * h) as usize];
    for y in 8..24 {
        for x in 8..24 {
            pixels[y * w as usize + x] = 255;
        }
    }
    let input = ctx.texture_gray8(&pixels, w, h).unwrap();
    let output = ctx.texture_output_gray8(w, h).unwrap();

    let cfg = vx_vision::kernels::morphology::MorphConfig { radius_x: 1, radius_y: 1 };
    morph.erode(&ctx, &input, &output, &cfg).unwrap();

    let result = output.read_gray8();
    let white_before = pixels.iter().filter(|&&v| v > 128).count();
    let white_after = result.iter().filter(|&&v| v > 128).count();
    assert!(white_after < white_before, "Erosion should shrink white region");
}

#[test]
fn morphology_dilate_grows_white() {
    let ctx = Context::new().unwrap();
    let morph = vx_vision::kernels::morphology::Morphology::new(&ctx).unwrap();

    let w = 32u32;
    let h = 32u32;
    // Small white dot
    let mut pixels = vec![0u8; (w * h) as usize];
    pixels[16 * 32 + 16] = 255;
    let input = ctx.texture_gray8(&pixels, w, h).unwrap();
    let output = ctx.texture_output_gray8(w, h).unwrap();

    let cfg = vx_vision::kernels::morphology::MorphConfig { radius_x: 2, radius_y: 2 };
    morph.dilate(&ctx, &input, &output, &cfg).unwrap();

    let result = output.read_gray8();
    let white_before = pixels.iter().filter(|&&v| v > 128).count();
    let white_after = result.iter().filter(|&&v| v > 128).count();
    assert!(white_after > white_before, "Dilation should grow white region");
}

// ── Bilateral Filter ────────────────────────────────────────────────

#[test]
fn bilateral_filter_preserves_edges() {
    let ctx = Context::new().unwrap();
    let bilateral = vx_vision::kernels::bilateral::BilateralFilter::new(&ctx).unwrap();

    let w = 64u32;
    let h = 64u32;
    // Sharp edge: left half black, right half white
    let pixels: Vec<u8> = (0..(w * h)).map(|i| {
        if (i % w) < w / 2 { 0 } else { 255 }
    }).collect();
    let input = ctx.texture_gray8(&pixels, w, h).unwrap();
    let output = ctx.texture_output_gray8(w, h).unwrap();

    let cfg = vx_vision::kernels::bilateral::BilateralConfig {
        radius: 5, sigma_spatial: 10.0, sigma_range: 0.05,
    };
    bilateral.apply(&ctx, &input, &output, &cfg).unwrap();

    let result = output.read_gray8();
    // Edge should be preserved: pixels far from edge should stay near 0 or 255
    let left_mean: f64 = result.iter().take((w / 4) as usize)
        .map(|&v| v as f64).sum::<f64>() / (w / 4) as f64;
    let right_center = (w * h / 2 + w * 3 / 4) as usize;
    let right_val = result[right_center] as f64;
    assert!(left_mean < 30.0, "Left side should stay dark: {}", left_mean);
    assert!(right_val > 220.0, "Right side should stay bright: {}", right_val);
}

// ── Pyramid ─────────────────────────────────────────────────────────

#[test]
fn pyramid_builds_levels() {
    let ctx = Context::new().unwrap();
    let pyr = vx_vision::kernels::pyramid::PyramidBuilder::new(&ctx).unwrap();

    let w = 128u32;
    let h = 128u32;
    let pixels = make_gradient_image(w, h);
    let input = ctx.texture_gray8(&pixels, w, h).unwrap();

    // build(n_levels=4) produces 3 downsampled levels (excludes the input)
    let levels = pyr.build(&ctx, &input, 4).unwrap();
    assert_eq!(levels.len(), 3);
    assert_eq!(levels[0].width(), 64);
    assert_eq!(levels[0].height(), 64);
    assert_eq!(levels[1].width(), 32);
    assert_eq!(levels[1].height(), 32);
    assert_eq!(levels[2].width(), 16);
    assert_eq!(levels[2].height(), 16);
}

// ── Resize ──────────────────────────────────────────────────────────

#[test]
fn resize_downscale() {
    let ctx = Context::new().unwrap();
    let resize = vx_vision::kernels::resize::ImageResize::new(&ctx).unwrap();

    let w = 64u32;
    let h = 64u32;
    let pixels = make_gradient_image(w, h);
    let input = ctx.texture_gray8(&pixels, w, h).unwrap();
    let output = ctx.texture_output_gray8(32, 32).unwrap();

    resize.apply(&ctx, &input, &output).unwrap();

    let result = output.read_gray8();
    assert_eq!(result.len(), 32 * 32);
    // Should still have gradient characteristics
    assert!(result[0] < result[31], "Gradient should be preserved after resize");
}

// ── Connected Components ────────────────────────────────────────────

#[test]
fn ccl_separate_blobs() {
    let ctx = Context::new().unwrap();
    let ccl = vx_vision::kernels::connected::ConnectedComponents::new(&ctx).unwrap();

    let w = 32u32;
    let h = 32u32;
    // Two separate white squares
    let mut pixels = vec![0u8; (w * h) as usize];
    for y in 2..8 {
        for x in 2..8 {
            pixels[y * w as usize + x] = 255;
        }
    }
    for y in 20..28 {
        for x in 20..28 {
            pixels[y * w as usize + x] = 255;
        }
    }
    let input = ctx.texture_gray8(&pixels, w, h).unwrap();

    let cfg = vx_vision::kernels::connected::CCLConfig {
        threshold: 0.5,
        max_iterations: 64,
    };
    let result = ccl.label(&ctx, &input, &cfg).unwrap();

    assert_eq!(result.n_components, 2, "Should find 2 components, found {}", result.n_components);
    assert!(result.iterations > 0, "Should take at least 1 iteration");
}

#[test]
fn ccl_uniform_black_no_components() {
    let ctx = Context::new().unwrap();
    let ccl = vx_vision::kernels::connected::ConnectedComponents::new(&ctx).unwrap();

    let w = 16u32;
    let h = 16u32;
    let pixels = make_black_image(w, h);
    let input = ctx.texture_gray8(&pixels, w, h).unwrap();

    let cfg = vx_vision::kernels::connected::CCLConfig::default();
    let result = ccl.label(&ctx, &input, &cfg).unwrap();

    assert_eq!(result.n_components, 0, "Black image should have 0 components");
}

// ── Distance Transform ──────────────────────────────────────────────

#[test]
fn distance_transform_single_seed() {
    let ctx = Context::new().unwrap();
    let dt = vx_vision::kernels::distance::DistanceTransform::new(&ctx).unwrap();

    let w = 32u32;
    let h = 32u32;
    // Single seed pixel in center
    let mut pixels = vec![255u8; (w * h) as usize];
    pixels[16 * 32 + 16] = 0; // seed
    let input = ctx.texture_gray8(&pixels, w, h).unwrap();

    let cfg = vx_vision::kernels::distance::DistanceConfig { threshold: 0.5 };
    let result = dt.compute(&ctx, &input, &cfg).unwrap();

    let distances = result.read_r32float();
    // Distance at seed should be 0
    assert!((distances[16 * 32 + 16]) < 0.01, "Seed distance should be ~0");
    // Distance at corner should be > 0
    assert!(distances[0] > 10.0, "Corner should be far from seed");
}

// ── Canny ───────────────────────────────────────────────────────────

#[test]
fn canny_detects_edges_in_checkerboard() {
    let ctx = Context::new().unwrap();
    let canny = vx_vision::kernels::canny::CannyDetector::new(&ctx).unwrap();

    let w = 64u32;
    let h = 64u32;
    let pixels = make_checkerboard(w, h, 16);
    let input = ctx.texture_gray8(&pixels, w, h).unwrap();

    let cfg = vx_vision::kernels::canny::CannyConfig {
        low_threshold: 0.04,
        high_threshold: 0.12,
        blur_sigma: 1.0,
        blur_radius: 3,
    };
    let edges = canny.detect(&ctx, &input, &cfg).unwrap();
    let edge_data = edges.read_r32float();
    let edge_count = edge_data.iter().filter(|&&v| v > 0.5).count();

    assert!(edge_count > 0, "Canny should detect edges in checkerboard");
    assert!(edge_count < (w * h) as usize / 2, "Too many edge pixels");
}

// ── Integral Image ──────────────────────────────────────────────────

#[test]
fn integral_image_sum() {
    let ctx = Context::new().unwrap();
    let integral = vx_vision::kernels::integral::IntegralImage::new(&ctx).unwrap();

    let w = 8u32;
    let h = 8u32;
    let pixels = make_white_image(w, h);
    let input = ctx.texture_gray8(&pixels, w, h).unwrap();

    let result = integral.compute(&ctx, &input).unwrap();
    let data = result.read_r32float();

    // Bottom-right should be sum of all pixels = w*h * (255/255) = w*h
    let bottom_right = data[(h - 1) as usize * w as usize + (w - 1) as usize];
    let expected = (w * h) as f32; // each pixel = 1.0 (normalized)
    assert!((bottom_right - expected).abs() < 1.0,
        "Bottom-right integral should be ~{}, got {}", expected, bottom_right);
}

// ── Hough Lines ─────────────────────────────────────────────────────

#[test]
fn hough_detects_horizontal_line() {
    let ctx = Context::new().unwrap();
    let hough = vx_vision::kernels::hough::HoughLines::new(&ctx).unwrap();

    let w = 64u32;
    let h = 64u32;
    // Horizontal white line across middle
    let mut pixels = vec![0u8; (w * h) as usize];
    for x in 0..w as usize {
        pixels[32 * w as usize + x] = 255;
    }
    let input = ctx.texture_gray8(&pixels, w, h).unwrap();

    let cfg = vx_vision::kernels::hough::HoughConfig {
        n_theta: 180,
        edge_threshold: 0.5,
        vote_threshold: 30,
        max_lines: 16,
        nms_radius: 5,
    };
    let lines = hough.detect(&ctx, &input, &cfg).unwrap();

    assert!(!lines.is_empty(), "Should detect at least one line");
    // Horizontal line → theta ≈ 90° (π/2)
    let has_horizontal = lines.iter().any(|l| {
        let deg = l.theta.to_degrees();
        (deg - 90.0).abs() < 10.0
    });
    assert!(has_horizontal, "Should detect horizontal line near 90°");
}

// ── Template Match ──────────────────────────────────────────────────

#[test]
fn template_match_finds_self() {
    let ctx = Context::new().unwrap();
    let tm = vx_vision::kernels::template_match::TemplateMatcher::new(&ctx).unwrap();

    let w = 64u32;
    let h = 64u32;
    // Create a gray image with a distinctive gradient patch at (20,20)
    let mut pixels = vec![128u8; (w * h) as usize];
    let pw = 8u32;
    let ph = 8u32;
    for dy in 0..ph {
        for dx in 0..pw {
            // Non-uniform patch: gradient pattern so NCC has non-zero variance
            pixels[((20 + dy) * w + (20 + dx)) as usize] = (dy * 32 + dx * 16) as u8;
        }
    }
    let image = ctx.texture_gray8(&pixels, w, h).unwrap();

    // Template is the same gradient patch
    let mut patch = vec![0u8; (pw * ph) as usize];
    for dy in 0..ph {
        for dx in 0..pw {
            patch[(dy * pw + dx) as usize] = (dy * 32 + dx * 16) as u8;
        }
    }
    let template = ctx.texture_gray8(&patch, pw, ph).unwrap();

    let result = tm.match_template(&ctx, &image, &template).unwrap();

    // Allow small offset tolerance for NCC edge effects
    let dx = (result.best_x as i32 - 20).unsigned_abs();
    let dy_off = (result.best_y as i32 - 20).unsigned_abs();
    assert!(dx <= 2 && dy_off <= 2,
        "Should find template near (20,20), got ({}, {})", result.best_x, result.best_y);
}

// ── Brute-Force Matcher ─────────────────────────────────────────────

#[test]
fn matcher_identical_descriptors() {
    let ctx = Context::new().unwrap();
    let matcher = vx_vision::kernels::matcher::BruteMatcher::new(&ctx).unwrap();

    // Two identical sets of 4 descriptors (8 u32s each)
    let desc: Vec<u32> = vec![
        0xAAAAAAAA, 0x55555555, 0x12345678, 0x9ABCDEF0,
        0x11111111, 0x22222222, 0x33333333, 0x44444444,
        0xFFFFFFFF, 0x00000000, 0xDEADBEEF, 0xCAFEBABE,
        0x01010101, 0x02020202, 0x03030303, 0x04040404,
        0x55555555, 0x66666666, 0x77777777, 0x88888888,
        0x99999999, 0xAAAAAAAA, 0xBBBBBBBB, 0xCCCCCCCC,
        0xDDDDDDDD, 0xEEEEEEEE, 0x11223344, 0x55667788,
        0x99AABBCC, 0xDDEEFF00, 0x11111111, 0x22222222,
    ];

    let cfg = vx_vision::kernels::matcher::MatchConfig {
        max_hamming: 256,
        ratio_thresh: 0.99,
    };
    let matches = matcher.match_descriptors(&ctx, &desc, &desc, &cfg).unwrap();

    // Should find matches (identical descriptors → distance 0)
    assert!(!matches.is_empty(), "Should match identical descriptors");
    for m in &matches {
        assert_eq!(m.distance, 0, "Identical descriptors should have distance 0");
    }
}

// ── Pipeline ────────────────────────────────────────────────────────

#[test]
fn pipeline_commit_and_wait() {
    let ctx = Context::new().unwrap();
    let blur = vx_vision::kernels::gaussian::GaussianBlur::new(&ctx).unwrap();

    let w = 32u32;
    let h = 32u32;
    let pixels = make_gradient_image(w, h);
    let input = ctx.texture_gray8(&pixels, w, h).unwrap();
    let output = ctx.texture_output_gray8(w, h).unwrap();

    let pipe = vx_vision::Pipeline::begin(&ctx).unwrap();
    let cfg = vx_vision::kernels::gaussian::GaussianConfig { sigma: 1.0, radius: 3 };
    let _state = blur.encode(&ctx, pipe.cmd_buf(), &input, &output, &cfg).unwrap();
    let _retained = pipe.commit_and_wait();

    let result = output.read_gray8();
    assert_eq!(result.len(), (w * h) as usize);
}
