use vx_vision::Context;

// Helper

fn make_gradient_image(w: u32, h: u32) -> Vec<u8> {
    (0..(w * h))
        .map(|i| {
            let x = i % w;
            ((x as f32 / w as f32) * 255.0) as u8
        })
        .collect()
}

fn make_checkerboard(w: u32, h: u32, block: u32) -> Vec<u8> {
    (0..(w * h))
        .map(|i| {
            let x = (i % w) / block;
            let y = (i / w) / block;
            if (x + y).is_multiple_of(2) {
                255u8
            } else {
                0u8
            }
        })
        .collect()
}

fn make_white_image(w: u32, h: u32) -> Vec<u8> {
    vec![255u8; (w * h) as usize]
}

fn make_black_image(w: u32, h: u32) -> Vec<u8> {
    vec![0u8; (w * h) as usize]
}

// Gaussian Blur

#[test]
fn gaussian_blur_reduces_noise() {
    let ctx = Context::new().unwrap();
    let blur = vx_vision::kernels::gaussian::GaussianBlur::new(&ctx).unwrap();

    let w = 64u32;
    let h = 64u32;
    let noisy: Vec<u8> = (0..(w * h))
        .map(|i| if i % 2 == 0 { 255 } else { 0 })
        .collect();
    let input = ctx.texture_gray8(&noisy, w, h).unwrap();
    let output = ctx.texture_output_gray8(w, h).unwrap();

    let mut cfg = vx_vision::kernels::gaussian::GaussianConfig::default();
    cfg.sigma = 2.0;
    cfg.radius = 6;
    blur.apply(&ctx, &input, &output, &cfg).unwrap();

    let result = output.read_gray8();
    let mean: f64 = result.iter().map(|&v| v as f64).sum::<f64>() / result.len() as f64;
    assert!((mean - 127.5).abs() < 20.0, "Mean after blur: {}", mean);

    let variance: f64 = result
        .iter()
        .map(|&v| {
            let d = v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / result.len() as f64;
    let orig_variance: f64 = noisy
        .iter()
        .map(|&v| {
            let d = v as f64 - 127.5;
            d * d
        })
        .sum::<f64>()
        / noisy.len() as f64;
    assert!(
        variance < orig_variance * 0.5,
        "Blur didn't reduce variance enough: {} vs {}",
        variance,
        orig_variance
    );
}

// Sobel

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

    let avg_gx: f32 = gx.iter().sum::<f32>() / gx.len() as f32;
    let avg_gy: f32 = gy.iter().map(|v| v.abs()).sum::<f32>() / gy.len() as f32;

    assert!(
        avg_gx.abs() > avg_gy,
        "Should detect horizontal gradient: gx={} gy={}",
        avg_gx,
        avg_gy
    );

    let max_mag = mag.iter().cloned().fold(0.0f32, f32::max);
    assert!(max_mag > 0.0, "Magnitude should be positive");
}

// FAST Corner Detection

#[test]
fn fast_detects_corners_in_checkerboard() {
    let ctx = Context::new().unwrap();
    let fast = vx_vision::kernels::fast::FastDetector::new(&ctx).unwrap();

    let w = 64u32;
    let h = 64u32;
    // Bright rectangle on black: FAST-9's Bresenham circle straddles the edges.
    let mut pixels = vec![0u8; (w * h) as usize];
    for y in 16..48u32 {
        for x in 16..48u32 {
            pixels[(y * w + x) as usize] = 255;
        }
    }
    let input = ctx.texture_gray8(&pixels, w, h).unwrap();

    let cfg = vx_vision::kernels::fast::FastDetectConfig::new(20, 4096);
    let result = fast.detect(&ctx, &input, &cfg).unwrap();

    assert!(
        !result.corners.is_empty(),
        "FAST should detect corners on rectangle edges (got 0)"
    );
}

#[test]
fn fast_no_corners_in_uniform() {
    let ctx = Context::new().unwrap();
    let fast = vx_vision::kernels::fast::FastDetector::new(&ctx).unwrap();

    let w = 64u32;
    let h = 64u32;
    let pixels = make_white_image(w, h);
    let input = ctx.texture_gray8(&pixels, w, h).unwrap();

    let cfg = vx_vision::kernels::fast::FastDetectConfig::new(20, 1024);
    let result = fast.detect(&ctx, &input, &cfg).unwrap();
    assert_eq!(
        result.corners.len(),
        0,
        "Uniform image should have no corners"
    );
}

// Histogram

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
    let pixels: Vec<u8> = (0..(w * h)).map(|i| 100 + (i % 51) as u8).collect();
    let input = ctx.texture_gray8(&pixels, w, h).unwrap();
    let output = ctx.texture_output_gray8(w, h).unwrap();

    hist.equalize(&ctx, &input, &output).unwrap();

    let result = output.read_gray8();
    let min_val = *result.iter().min().unwrap();
    let max_val = *result.iter().max().unwrap();
    let range = max_val as i32 - min_val as i32;

    assert!(
        range > 100,
        "Equalization should expand range: got {}",
        range
    );
}

// Threshold

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
        assert!(
            val == 0 || val == 255,
            "Binary should produce only 0 or 255, got {}",
            val
        );
    }
}

#[test]
fn threshold_otsu() {
    let ctx = Context::new().unwrap();
    let thresh = vx_vision::kernels::threshold::Threshold::new(&ctx).unwrap();

    let w = 64u32;
    let h = 64u32;
    // Bimodal distribution so Otsu has a clear split point
    let mut pixels = vec![0u8; (w * h) as usize];
    for pixel in pixels.iter_mut().take((w * h / 2) as usize) {
        *pixel = 50;
    }
    for pixel in pixels.iter_mut().skip((w * h / 2) as usize) {
        *pixel = 200;
    }
    let input = ctx.texture_gray8(&pixels, w, h).unwrap();
    let output = ctx.texture_output_gray8(w, h).unwrap();

    let otsu_val = thresh.otsu(&ctx, &input, &output).unwrap();

    // Normalized [0,1]; should fall between modes 50/255 and 200/255
    assert!(
        otsu_val > 0.15 && otsu_val < 0.85,
        "Otsu threshold should be between modes: got {}",
        otsu_val
    );

    let result = output.read_gray8();
    for &val in &result {
        assert!(val == 0 || val == 255, "Otsu output should be binary");
    }
}

// Morphology

#[test]
fn morphology_erode_shrinks_white() {
    let ctx = Context::new().unwrap();
    let morph = vx_vision::kernels::morphology::Morphology::new(&ctx).unwrap();

    let w = 32u32;
    let h = 32u32;
    let mut pixels = vec![0u8; (w * h) as usize];
    for y in 8..24 {
        for x in 8..24 {
            pixels[y * w as usize + x] = 255;
        }
    }
    let input = ctx.texture_gray8(&pixels, w, h).unwrap();
    let output = ctx.texture_output_gray8(w, h).unwrap();

    let cfg = vx_vision::kernels::morphology::MorphConfig::new(1, 1);
    morph.erode(&ctx, &input, &output, &cfg).unwrap();

    let result = output.read_gray8();
    let white_before = pixels.iter().filter(|&&v| v > 128).count();
    let white_after = result.iter().filter(|&&v| v > 128).count();
    assert!(
        white_after < white_before,
        "Erosion should shrink white region"
    );
}

#[test]
fn morphology_dilate_grows_white() {
    let ctx = Context::new().unwrap();
    let morph = vx_vision::kernels::morphology::Morphology::new(&ctx).unwrap();

    let w = 32u32;
    let h = 32u32;
    let mut pixels = vec![0u8; (w * h) as usize];
    pixels[16 * 32 + 16] = 255;
    let input = ctx.texture_gray8(&pixels, w, h).unwrap();
    let output = ctx.texture_output_gray8(w, h).unwrap();

    let cfg = vx_vision::kernels::morphology::MorphConfig::new(2, 2);
    morph.dilate(&ctx, &input, &output, &cfg).unwrap();

    let result = output.read_gray8();
    let white_before = pixels.iter().filter(|&&v| v > 128).count();
    let white_after = result.iter().filter(|&&v| v > 128).count();
    assert!(
        white_after > white_before,
        "Dilation should grow white region"
    );
}

// Bilateral Filter

#[test]
fn bilateral_filter_preserves_edges() {
    let ctx = Context::new().unwrap();
    let bilateral = vx_vision::kernels::bilateral::BilateralFilter::new(&ctx).unwrap();

    let w = 64u32;
    let h = 64u32;
    let pixels: Vec<u8> = (0..(w * h))
        .map(|i| if (i % w) < w / 2 { 0 } else { 255 })
        .collect();
    let input = ctx.texture_gray8(&pixels, w, h).unwrap();
    let output = ctx.texture_output_gray8(w, h).unwrap();

    let cfg = vx_vision::kernels::bilateral::BilateralConfig::new(5, 10.0, 0.05);
    bilateral.apply(&ctx, &input, &output, &cfg).unwrap();

    let result = output.read_gray8();
    let left_mean: f64 = result
        .iter()
        .take((w / 4) as usize)
        .map(|&v| v as f64)
        .sum::<f64>()
        / (w / 4) as f64;
    let right_center = (w * h / 2 + w * 3 / 4) as usize;
    let right_val = result[right_center] as f64;
    assert!(
        left_mean < 30.0,
        "Left side should stay dark: {}",
        left_mean
    );
    assert!(
        right_val > 220.0,
        "Right side should stay bright: {}",
        right_val
    );
}

// Pyramid

#[test]
fn pyramid_builds_levels() {
    let ctx = Context::new().unwrap();
    let pyr = vx_vision::kernels::pyramid::PyramidBuilder::new(&ctx).unwrap();

    let w = 128u32;
    let h = 128u32;
    let pixels = make_gradient_image(w, h);
    let input = ctx.texture_gray8(&pixels, w, h).unwrap();

    // n_levels=4 produces 3 downsampled levels (excludes input)
    let levels = pyr.build(&ctx, &input, 4).unwrap();
    assert_eq!(levels.len(), 3);
    assert_eq!(levels[0].width(), 64);
    assert_eq!(levels[0].height(), 64);
    assert_eq!(levels[1].width(), 32);
    assert_eq!(levels[1].height(), 32);
    assert_eq!(levels[2].width(), 16);
    assert_eq!(levels[2].height(), 16);
}

// Resize

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
    assert!(
        result[0] < result[31],
        "Gradient should be preserved after resize"
    );
}

// Connected Components

#[test]
fn ccl_separate_blobs() {
    let ctx = Context::new().unwrap();
    let ccl = vx_vision::kernels::connected::ConnectedComponents::new(&ctx).unwrap();

    let w = 32u32;
    let h = 32u32;
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

    let mut cfg = vx_vision::kernels::connected::CCLConfig::default();
    cfg.threshold = 0.5;
    cfg.max_iterations = 64;
    let result = ccl.label(&ctx, &input, &cfg).unwrap();

    assert_eq!(
        result.n_components, 2,
        "Should find 2 components, found {}",
        result.n_components
    );
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

    assert_eq!(
        result.n_components, 0,
        "Black image should have 0 components"
    );
}

// Distance Transform

#[test]
fn distance_transform_single_seed() {
    let ctx = Context::new().unwrap();
    let dt = vx_vision::kernels::distance::DistanceTransform::new(&ctx).unwrap();

    let w = 32u32;
    let h = 32u32;
    let mut pixels = vec![255u8; (w * h) as usize];
    pixels[16 * 32 + 16] = 0;
    let input = ctx.texture_gray8(&pixels, w, h).unwrap();

    let mut cfg = vx_vision::kernels::distance::DistanceConfig::default();
    cfg.threshold = 0.5;
    let result = dt.compute(&ctx, &input, &cfg).unwrap();

    let distances = result.read_r32float();
    assert!(
        (distances[16 * 32 + 16]) < 0.01,
        "Seed distance should be ~0"
    );
    assert!(distances[0] > 10.0, "Corner should be far from seed");
}

// Canny

#[test]
fn canny_detects_edges_in_checkerboard() {
    let ctx = Context::new().unwrap();
    let canny = vx_vision::kernels::canny::CannyDetector::new(&ctx).unwrap();

    let w = 64u32;
    let h = 64u32;
    let pixels = make_checkerboard(w, h, 16);
    let input = ctx.texture_gray8(&pixels, w, h).unwrap();

    let mut cfg = vx_vision::kernels::canny::CannyConfig::default();
    cfg.low_threshold = 0.04;
    cfg.high_threshold = 0.12;
    cfg.blur_sigma = 1.0;
    cfg.blur_radius = 3;
    let edges = canny.detect(&ctx, &input, &cfg).unwrap();
    let edge_data = edges.read_r32float();
    let edge_count = edge_data.iter().filter(|&&v| v > 0.5).count();

    assert!(edge_count > 0, "Canny should detect edges in checkerboard");
    assert!(edge_count < (w * h) as usize / 2, "Too many edge pixels");
}

// Integral Image

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

    let bottom_right = data[(h - 1) as usize * w as usize + (w - 1) as usize];
    // Each pixel normalizes to 1.0, so sum = w*h
    let expected = (w * h) as f32;
    assert!(
        (bottom_right - expected).abs() < 1.0,
        "Bottom-right integral should be ~{}, got {}",
        expected,
        bottom_right
    );
}

// Hough Lines

#[test]
fn hough_detects_horizontal_line() {
    let ctx = Context::new().unwrap();
    let hough = vx_vision::kernels::hough::HoughLines::new(&ctx).unwrap();

    let w = 64u32;
    let h = 64u32;
    let mut pixels = vec![0u8; (w * h) as usize];
    for x in 0..w as usize {
        pixels[32 * w as usize + x] = 255;
    }
    let input = ctx.texture_gray8(&pixels, w, h).unwrap();

    let mut cfg = vx_vision::kernels::hough::HoughConfig::default();
    cfg.n_theta = 180;
    cfg.edge_threshold = 0.5;
    cfg.vote_threshold = 30;
    cfg.max_lines = 16;
    cfg.nms_radius = 5;
    let lines = hough.detect(&ctx, &input, &cfg).unwrap();

    assert!(!lines.is_empty(), "Should detect at least one line");
    let has_horizontal = lines.iter().any(|l| {
        let deg = l.theta.to_degrees();
        (deg - 90.0).abs() < 10.0
    });
    assert!(has_horizontal, "Should detect horizontal line near 90°");
}

// Template Match

#[test]
fn template_match_finds_self() {
    let ctx = Context::new().unwrap();
    let tm = vx_vision::kernels::template_match::TemplateMatcher::new(&ctx).unwrap();

    let w = 64u32;
    let h = 64u32;
    let mut pixels = vec![128u8; (w * h) as usize];
    let pw = 8u32;
    let ph = 8u32;
    // Gradient patch so NCC has non-zero variance
    for dy in 0..ph {
        for dx in 0..pw {
            pixels[((20 + dy) * w + (20 + dx)) as usize] = (dy * 32 + dx * 16) as u8;
        }
    }
    let image = ctx.texture_gray8(&pixels, w, h).unwrap();

    let mut patch = vec![0u8; (pw * ph) as usize];
    for dy in 0..ph {
        for dx in 0..pw {
            patch[(dy * pw + dx) as usize] = (dy * 32 + dx * 16) as u8;
        }
    }
    let template = ctx.texture_gray8(&patch, pw, ph).unwrap();

    let result = tm.match_template(&ctx, &image, &template).unwrap();

    let dx = (result.best_x as i32 - 20).unsigned_abs();
    let dy_off = (result.best_y as i32 - 20).unsigned_abs();
    assert!(
        dx <= 2 && dy_off <= 2,
        "Should find template near (20,20), got ({}, {})",
        result.best_x,
        result.best_y
    );
}

// Brute-Force Matcher

#[test]
fn matcher_identical_descriptors() {
    let ctx = Context::new().unwrap();
    let matcher = vx_vision::kernels::matcher::BruteMatcher::new(&ctx).unwrap();

    let desc: Vec<u32> = vec![
        0xAAAAAAAA, 0x55555555, 0x12345678, 0x9ABCDEF0, 0x11111111, 0x22222222, 0x33333333,
        0x44444444, 0xFFFFFFFF, 0x00000000, 0xDEADBEEF, 0xCAFEBABE, 0x01010101, 0x02020202,
        0x03030303, 0x04040404, 0x55555555, 0x66666666, 0x77777777, 0x88888888, 0x99999999,
        0xAAAAAAAA, 0xBBBBBBBB, 0xCCCCCCCC, 0xDDDDDDDD, 0xEEEEEEEE, 0x11223344, 0x55667788,
        0x99AABBCC, 0xDDEEFF00, 0x11111111, 0x22222222,
    ];

    let mut cfg = vx_vision::kernels::matcher::MatchConfig::default();
    cfg.max_hamming = 256;
    cfg.ratio_thresh = 0.99;
    let matches = matcher.match_descriptors(&ctx, &desc, &desc, &cfg).unwrap();

    assert!(!matches.is_empty(), "Should match identical descriptors");
    for m in &matches {
        assert_eq!(
            m.distance, 0,
            "Identical descriptors should have distance 0"
        );
    }
}

// Pipeline

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
    let mut cfg = vx_vision::kernels::gaussian::GaussianConfig::default();
    cfg.sigma = 1.0;
    cfg.radius = 3;
    let _state = blur
        .encode(&ctx, pipe.cmd_buf(), &input, &output, &cfg)
        .unwrap();
    let _retained = pipe.commit_and_wait();

    let result = output.read_gray8();
    assert_eq!(result.len(), (w * h) as usize);
}

#[test]
fn pipeline_five_kernel_chain() {
    use vx_vision::kernels::bilateral::{BilateralConfig, BilateralFilter};
    use vx_vision::kernels::gaussian::{GaussianBlur, GaussianConfig};
    use vx_vision::kernels::morphology::{MorphConfig, Morphology};
    use vx_vision::kernels::threshold::Threshold;

    let ctx = Context::new().unwrap();
    let gauss = GaussianBlur::new(&ctx).unwrap();
    let bilateral = BilateralFilter::new(&ctx).unwrap();
    let morph = Morphology::new(&ctx).unwrap();
    let thresh = Threshold::new(&ctx).unwrap();

    let w = 64u32;
    let h = 64u32;
    let pixels = make_gradient_image(w, h);
    let input = ctx.texture_gray8(&pixels, w, h).unwrap();

    let inter1 = ctx.texture_intermediate_gray8(w, h).unwrap();
    let inter2 = ctx.texture_intermediate_gray8(w, h).unwrap();
    let inter3 = ctx.texture_intermediate_gray8(w, h).unwrap();
    let inter4 = ctx.texture_intermediate_gray8(w, h).unwrap();
    let output = ctx.texture_output_gray8(w, h).unwrap();

    let pipe = vx_vision::Pipeline::begin(&ctx).unwrap();
    let cmd = pipe.cmd_buf();

    let mut gauss_cfg = GaussianConfig::default();
    gauss_cfg.sigma = 1.0;
    gauss_cfg.radius = 2;
    let _gauss_state = gauss
        .encode(&ctx, cmd, &input, &inter1, &gauss_cfg)
        .unwrap();

    let bilateral_cfg = BilateralConfig::default();
    bilateral
        .encode(cmd, &inter1, &inter2, &bilateral_cfg)
        .unwrap();

    let morph_cfg = MorphConfig::default();
    morph
        .encode_dilate(cmd, &inter2, &inter3, &morph_cfg)
        .unwrap();

    morph
        .encode_erode(cmd, &inter3, &inter4, &morph_cfg)
        .unwrap();

    thresh
        .encode_binary(cmd, &inter4, &output, 0.5, false)
        .unwrap();

    let _retained = pipe.commit_and_wait();

    let result = output.read_gray8();
    assert_eq!(result.len(), (w * h) as usize);
    let nonzero = result.iter().filter(|&&v| v > 0).count();
    assert!(
        nonzero > 0,
        "threshold output should have some white pixels"
    );
    assert!(
        nonzero < result.len(),
        "threshold output should have some black pixels"
    );
}

// Multi-threaded dispatch

#[test]
fn multithreaded_dispatch() {
    use std::sync::Arc;
    use std::thread;
    use vx_vision::kernels::fast::{FastDetectConfig, FastDetector};
    use vx_vision::kernels::pyramid::PyramidBuilder;

    let ctx = Arc::new(Context::new().unwrap());

    let w = 64u32;
    let h = 64u32;
    let pixels = make_gradient_image(w, h);

    let ctx1 = Arc::clone(&ctx);
    let pixels1 = pixels.clone();
    let t1 = thread::spawn(move || {
        let input = ctx1.texture_gray8(&pixels1, w, h).unwrap();
        let pyr = PyramidBuilder::new(&ctx1).unwrap();
        let levels = pyr.build(&ctx1, &input, 3).unwrap();
        assert!(levels.len() >= 2, "pyramid should produce multiple levels");
    });

    let ctx2 = Arc::clone(&ctx);
    let pixels2 = pixels.clone();
    let t2 = thread::spawn(move || {
        let input = ctx2.texture_gray8(&pixels2, w, h).unwrap();
        let det = FastDetector::new(&ctx2).unwrap();
        let cfg = FastDetectConfig::default();
        let result = det.detect(&ctx2, &input, &cfg).unwrap();
        let _ = result;
    });

    t1.join().expect("pyramid thread panicked");
    t2.join().expect("FAST thread panicked");
}

// ── 3D Reconstruction: SGM Stereo ──

#[cfg(feature = "reconstruction")]
#[test]
fn sgm_stereo_produces_disparity() {
    let ctx = Context::new().unwrap();
    let sgm = vx_vision::kernels::sgm::SGMStereo::new(&ctx).unwrap();

    let w = 64u32;
    let h = 64u32;
    let shift = 5u32; // 5-pixel disparity

    // Create a textured pattern (checkerboard) that Census transform can match
    let left: Vec<u8> = (0..(w * h))
        .map(|i| {
            let x = i % w;
            let y = i / w;
            // Checkerboard with 4-pixel blocks gives Census transform good texture
            if ((x / 4) + (y / 4)) % 2 == 0 {
                200u8
            } else {
                50u8
            }
        })
        .collect();
    let right: Vec<u8> = (0..(w * h))
        .map(|i| {
            let x = i % w;
            let y = i / w;
            // Same pattern, shifted right (which means left image disparity = shift)
            let sx = if x >= shift { x - shift } else { 0 };
            if ((sx / 4) + (y / 4)) % 2 == 0 {
                200u8
            } else {
                50u8
            }
        })
        .collect();

    let left_tex = ctx.texture_gray8(&left, w, h).unwrap();
    let right_tex = ctx.texture_gray8(&right, w, h).unwrap();
    let output = ctx.texture_output_r32float(w, h).unwrap();

    let config = vx_vision::kernels::sgm::SGMStereoConfig::new(32);

    sgm.compute_disparity(&ctx, &left_tex, &right_tex, &output, &config)
        .unwrap();

    let result = output.read_r32float();
    // The pipeline should run without errors and produce a disparity map.
    // With a synthetic shifted checkerboard, most interior pixels should
    // have a non-zero disparity value from the cost volume.
    assert_eq!(result.len(), (w * h) as usize);
    // At minimum, verify the pipeline executed and produced valid float output
    let has_finite = result.iter().any(|&v| v.is_finite());
    assert!(has_finite, "SGM output should contain finite values");
}

// ── 3D Reconstruction: Depth Filter ──

#[cfg(feature = "reconstruction")]
#[test]
fn depth_bilateral_preserves_edges() {
    let ctx = Context::new().unwrap();
    let filter = vx_vision::kernels::depth_filter::DepthFilter::new(&ctx).unwrap();

    let w = 32u32;
    let h = 32u32;
    // Create a depth map with a sharp edge: left half = 1.0, right half = 5.0
    let depth: Vec<f32> = (0..(w * h))
        .map(|i| if (i % w) < w / 2 { 1.0 } else { 5.0 })
        .collect();

    let input = ctx.texture_r32float(&depth, w, h).unwrap();
    let output = ctx.texture_output_r32float(w, h).unwrap();

    let config = vx_vision::kernels::depth_filter::DepthFilterConfig::new(2, 3.0, 0.1);

    filter
        .apply_bilateral(&ctx, &input, &output, &config)
        .unwrap();

    let result = output.read_r32float();
    // Center of left half should still be close to 1.0
    let left_center = result[(h / 2 * w + w / 4) as usize];
    let right_center = result[(h / 2 * w + 3 * w / 4) as usize];

    assert!(
        (left_center - 1.0).abs() < 0.5,
        "left half center should be ~1.0, got {}",
        left_center
    );
    assert!(
        (right_center - 5.0).abs() < 0.5,
        "right half center should be ~5.0, got {}",
        right_center
    );
}

#[cfg(feature = "reconstruction")]
#[test]
fn depth_median_removes_noise() {
    let ctx = Context::new().unwrap();
    let filter = vx_vision::kernels::depth_filter::DepthFilter::new(&ctx).unwrap();

    let w = 32u32;
    let h = 32u32;
    // Create a uniform depth map with a few noisy pixels
    let mut depth: Vec<f32> = vec![2.0; (w * h) as usize];
    depth[100] = 99.0; // noise
    depth[200] = 0.5; // noise

    let input = ctx.texture_r32float(&depth, w, h).unwrap();
    let output = ctx.texture_output_r32float(w, h).unwrap();

    let config = vx_vision::kernels::depth_filter::DepthMedianConfig::default();

    filter.apply_median(&ctx, &input, &output, &config).unwrap();

    let result = output.read_r32float();
    // The noisy pixel should be smoothed out
    assert!(
        (result[100] - 2.0).abs() < 1.0,
        "noisy pixel should be closer to 2.0 after median, got {}",
        result[100]
    );
}

// ── 3D Reconstruction: Depth Inpaint ──

#[cfg(feature = "reconstruction")]
#[test]
fn depth_inpaint_fills_holes() {
    let ctx = Context::new().unwrap();
    let inpaint = vx_vision::kernels::depth_inpaint::DepthInpaint::new(&ctx).unwrap();

    let w = 32u32;
    let h = 32u32;
    // Create depth map with a hole in the center
    let depth: Vec<f32> = (0..(w * h))
        .map(|i| {
            let x = (i % w) as i32 - w as i32 / 2;
            let y = (i / w) as i32 - h as i32 / 2;
            if x * x + y * y < 25 {
                0.0 // hole
            } else {
                3.0 // valid depth
            }
        })
        .collect();

    let input = ctx.texture_r32float(&depth, w, h).unwrap();
    // Need intermediate usage for ping-pong
    let output = ctx.texture_intermediate_r32float(w, h).unwrap();

    let mut config = vx_vision::kernels::depth_inpaint::DepthInpaintConfig::default();
    config.max_iterations = 4;

    inpaint.apply(&ctx, &input, &output, &config).unwrap();

    let result = output.read_r32float();
    // Center pixel (was hole) should now have some fill value
    let center = result[(h / 2 * w + w / 2) as usize];
    assert!(center > 0.0, "center hole should be filled, got {}", center);
}

// ── 3D Reconstruction: Depth-to-Point-Cloud ──

#[cfg(feature = "reconstruction")]
#[test]
fn depth_to_cloud_unprojects_correctly() {
    let ctx = Context::new().unwrap();
    let d2c = vx_vision::kernels::depth_to_cloud::DepthToCloud::new(&ctx).unwrap();

    let w = 32u32;
    let h = 32u32;
    let fx = 100.0f32;
    let fy = 100.0f32;
    let cx = 16.0f32;
    let cy = 16.0f32;

    // Create a flat depth map at depth = 2.0
    let depth: Vec<f32> = vec![2.0; (w * h) as usize];
    let depth_tex = ctx.texture_r32float(&depth, w, h).unwrap();

    let intrinsics = vx_vision::types_3d::CameraIntrinsics::new(fx, fy, cx, cy, w, h);
    let depth_map = vx_vision::types_3d::DepthMap::new(depth_tex, intrinsics, 0.1, 10.0).unwrap();

    let config = vx_vision::kernels::depth_to_cloud::DepthToCloudConfig::new(0.1, 10.0);

    let cloud = d2c.compute(&ctx, &depth_map, None, &config).unwrap();

    assert_eq!(
        cloud.len(),
        (w * h) as usize,
        "all pixels should produce points"
    );

    // All points should have Z = 2.0
    for p in &cloud.points {
        assert!(
            (p.position[2] - 2.0).abs() < 0.01,
            "Z should be 2.0, got {}",
            p.position[2]
        );
    }
}

// ── 3D Reconstruction: Depth Colorize ──

#[cfg(feature = "reconstruction")]
#[test]
fn depth_colorize_produces_rgba() {
    let ctx = Context::new().unwrap();
    let colorize = vx_vision::kernels::depth_colorize::DepthColorize::new(&ctx).unwrap();

    let w = 32u32;
    let h = 32u32;
    // Create a depth gradient from 1.0 to 5.0
    let depth: Vec<f32> = (0..(w * h))
        .map(|i| 1.0 + 4.0 * (i % w) as f32 / w as f32)
        .collect();

    let input = ctx.texture_r32float(&depth, w, h).unwrap();
    let output = ctx.texture_output_rgba8(w, h).unwrap();

    let config = vx_vision::kernels::depth_colorize::DepthColorizeConfig::new(1.0, 5.0);

    colorize.apply(&ctx, &input, &output, &config).unwrap();

    let result = output.read_rgba8();
    // Output should be non-zero (colored)
    let nonblack = result
        .chunks(4)
        .filter(|px| px[0] > 0 || px[1] > 0 || px[2] > 0)
        .count();
    assert!(
        nonblack > (w * h / 2) as usize,
        "most pixels should have color, got {} of {}",
        nonblack,
        w * h
    );
}

// ── 3D Reconstruction: Normal Estimation ──

#[cfg(feature = "reconstruction")]
#[test]
fn normal_estimation_organized() {
    let ctx = Context::new().unwrap();
    let estimator = vx_vision::kernels::normal_estimation::NormalEstimator::new(&ctx).unwrap();

    let w = 32u32;
    let h = 32u32;
    // Flat depth map at z=2.0 → normals should point toward camera (0, 0, -1)
    let depth: Vec<f32> = vec![2.0; (w * h) as usize];
    let input = ctx.texture_r32float(&depth, w, h).unwrap();
    let output = ctx.texture_output_rgba8(w, h).unwrap();

    estimator
        .compute_from_depth(&ctx, &input, &output, 100.0, 100.0, 16.0, 16.0)
        .unwrap();

    let result = output.read_rgba8();
    // Interior pixels should have valid normals (non-zero alpha)
    let center_idx = (h / 2 * w + w / 2) as usize * 4;
    // Normal map encodes as (n+1)/2, so (0,0,-1) → (0.5, 0.5, 0.0)
    assert!(result[center_idx + 3] > 0, "alpha should be non-zero");
}

#[cfg(feature = "reconstruction")]
#[test]
fn normal_estimation_unorganized_plane() {
    use vx_vision::types_3d::{Point3D, PointCloud};

    let ctx = Context::new().unwrap();
    let estimator = vx_vision::kernels::normal_estimation::NormalEstimator::new(&ctx).unwrap();

    // Create a planar point cloud in the XY plane (z=0) with some spread
    let mut points = Vec::new();
    for y in 0..10 {
        for x in 0..10 {
            points.push(Point3D {
                position: [x as f32 * 0.1, y as f32 * 0.1, 0.0],
                ..Default::default()
            });
        }
    }
    let cloud = PointCloud { points };

    let mut config = vx_vision::kernels::normal_estimation::NormalEstimatorConfig::default();
    config.radius = 0.25;
    config.max_neighbors = 20;

    let normals = estimator.compute(&ctx, &cloud, &config).unwrap();
    assert_eq!(normals.len(), 100);

    // Most interior normals should point approximately in Z direction
    let center_normal = normals[55]; // an interior point
    let z_component = center_normal[2].abs();
    assert!(
        z_component > 0.5,
        "plane normal should be mostly in Z, got {:?}",
        center_normal
    );
}

// ── 3D Reconstruction: Outlier Filter ──

#[cfg(feature = "reconstruction")]
#[test]
fn outlier_filter_removes_distant_points() {
    use vx_vision::types_3d::{Point3D, PointCloud};

    let ctx = Context::new().unwrap();
    let filter = vx_vision::kernels::outlier_filter::OutlierFilter::new(&ctx).unwrap();

    // Create a tight cluster + one far outlier
    let mut points = Vec::new();
    for i in 0..50 {
        points.push(Point3D {
            position: [(i % 10) as f32 * 0.01, (i / 10) as f32 * 0.01, 0.0],
            ..Default::default()
        });
    }
    // Add outlier far away
    points.push(Point3D {
        position: [100.0, 100.0, 100.0],
        ..Default::default()
    });

    let cloud = PointCloud { points };
    let config = vx_vision::kernels::outlier_filter::OutlierFilterConfig::default();

    let filtered = filter.filter(&ctx, &cloud, &config).unwrap();

    assert!(
        filtered.len() < cloud.len(),
        "outlier should be removed: {} → {}",
        cloud.len(),
        filtered.len()
    );
    assert!(
        filtered.len() >= 49,
        "most cluster points should survive: {}",
        filtered.len()
    );
}

// ── 3D Reconstruction: Voxel Downsample ──

#[cfg(feature = "reconstruction")]
#[test]
fn voxel_downsample_reduces_points() {
    use vx_vision::types_3d::{Point3D, PointCloud};

    let ctx = Context::new().unwrap();
    let ds = vx_vision::kernels::voxel_downsample::VoxelDownsample::new(&ctx).unwrap();

    // Create a dense grid of points
    let mut points = Vec::new();
    for z in 0..10 {
        for y in 0..10 {
            for x in 0..10 {
                points.push(Point3D {
                    position: [x as f32 * 0.01, y as f32 * 0.01, z as f32 * 0.01],
                    ..Default::default()
                });
            }
        }
    }
    let cloud = PointCloud { points };
    assert_eq!(cloud.len(), 1000);

    // Voxel size of 0.05 should merge ~5×5×5 = 125 points per voxel
    let config = vx_vision::kernels::voxel_downsample::VoxelDownsampleConfig::new(0.05);

    let result = ds.downsample(&ctx, &cloud, &config).unwrap();

    assert!(
        result.len() < cloud.len(),
        "downsampling should reduce point count: {} → {}",
        cloud.len(),
        result.len()
    );
    assert!(result.len() > 0, "should have at least some output points");
}

// ── 3D Reconstruction: TSDF Volume Fusion ──

#[cfg(feature = "reconstruction")]
#[test]
fn tsdf_integrate_and_extract() {
    use vx_vision::kernels::tsdf::{TSDFConfig, TSDFVolume};
    use vx_vision::types_3d::{CameraExtrinsics, CameraIntrinsics, DepthMap};

    let ctx = Context::new().unwrap();

    let mut config = TSDFConfig::default();
    config.resolution = [64, 64, 64];
    config.voxel_size = 0.01;
    config.origin = [-0.32, -0.32, 0.0];
    config.truncation_distance = 0.03;

    let tsdf = TSDFVolume::new(&ctx, config).unwrap();

    let w = 32u32;
    let h = 32u32;
    let intrinsics = CameraIntrinsics::new(100.0, 100.0, 16.0, 16.0, w, h);

    // Create a depth map of a flat surface at z=0.32 (center of volume)
    let depth_data: Vec<f32> = vec![0.32; (w * h) as usize];
    let depth_tex = ctx.texture_r32float(&depth_data, w, h).unwrap();
    let depth_map = DepthMap::new(depth_tex, intrinsics, 0.1, 1.0).unwrap();

    let pose = CameraExtrinsics::identity();

    tsdf.integrate(&ctx, &depth_map, &pose).unwrap();

    // Extract surface points
    let cloud = tsdf.extract_cloud();
    assert!(
        cloud.len() > 0,
        "should extract some surface points after integration"
    );
}

#[cfg(feature = "reconstruction")]
#[test]
fn tsdf_raycast_produces_depth() {
    use vx_vision::kernels::tsdf::{TSDFConfig, TSDFVolume};
    use vx_vision::types_3d::{CameraExtrinsics, CameraIntrinsics, DepthMap};

    let ctx = Context::new().unwrap();

    let mut config = TSDFConfig::default();
    config.resolution = [64, 64, 64];
    config.voxel_size = 0.01;
    config.origin = [-0.32, -0.32, 0.0];
    config.truncation_distance = 0.03;

    let tsdf = TSDFVolume::new(&ctx, config).unwrap();

    let w = 32u32;
    let h = 32u32;
    let intrinsics = CameraIntrinsics::new(100.0, 100.0, 16.0, 16.0, w, h);

    // Integrate a flat surface
    let depth_data: Vec<f32> = vec![0.32; (w * h) as usize];
    let depth_tex = ctx.texture_r32float(&depth_data, w, h).unwrap();
    let depth_map = DepthMap::new(depth_tex, intrinsics, 0.1, 1.0).unwrap();
    let pose = CameraExtrinsics::identity();

    tsdf.integrate(&ctx, &depth_map, &pose).unwrap();

    // Raycast from same pose
    let (raycast_depth, _raycast_normal) = tsdf.raycast(&ctx, &pose, &intrinsics).unwrap();

    let depth_result = raycast_depth.read_r32float();
    // At least some rays should find the surface
    let valid = depth_result.iter().filter(|&&d| d > 0.0).count();
    assert!(valid > 0, "raycast should find some surface points");
}

// ── 3D Reconstruction: Marching Cubes ──

#[cfg(feature = "reconstruction")]
#[test]
fn marching_cubes_extracts_sphere_mesh() {
    let ctx = Context::new().unwrap();

    // Create a VoxelGrid with a sphere SDF
    let res = [32u32, 32, 32];
    let vs = 0.1f32;
    let origin = [-1.6f32, -1.6, -1.6];
    let mut grid = vx_vision::types_3d::VoxelGrid::from_context(&ctx, res, vs, origin).unwrap();

    // Write analytical sphere SDF: distance to center - radius
    let center = [0.0f32, 0.0, 0.0];
    let radius = 1.0f32;
    let total = res[0] as usize * res[1] as usize * res[2] as usize;
    let mut sdf_data = vec![1.0f32; total];

    for iz in 0..res[2] {
        for iy in 0..res[1] {
            for ix in 0..res[0] {
                let pos = grid.voxel_to_world(ix, iy, iz);
                let dx = pos[0] - center[0];
                let dy = pos[1] - center[1];
                let dz = pos[2] - center[2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt() - radius;
                let idx = grid.voxel_index(ix, iy, iz);
                sdf_data[idx] = dist;
            }
        }
    }
    grid.tsdf.write(&sdf_data);
    // Set weights > 0 so extraction works
    grid.weights.write(&vec![1.0f32; total]);

    let config = vx_vision::kernels::marching_cubes::MarchingCubesConfig::default();
    let mesh = vx_vision::kernels::marching_cubes::MarchingCubes::extract(&grid, &config);

    assert!(
        mesh.num_faces() > 10,
        "sphere mesh should have many faces, got {}",
        mesh.num_faces()
    );

    // Verify all vertices are approximately on the sphere surface
    for v in &mesh.vertices {
        let dx = v.position[0] - center[0];
        let dy = v.position[1] - center[1];
        let dz = v.position[2] - center[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        assert!(
            (dist - radius).abs() < vs * 2.0,
            "vertex at dist {} should be near radius {}",
            dist,
            radius
        );
    }
}

// ── 3D Reconstruction: Mesh Decimation ──

#[cfg(feature = "reconstruction")]
#[test]
fn mesh_decimate_reduces_faces() {
    use vx_vision::types_3d::{Mesh, Vertex3D};

    // Create a small mesh grid
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let n = 5;
    for y in 0..n {
        for x in 0..n {
            vertices.push(Vertex3D {
                position: [x as f32, y as f32, 0.0],
                ..Default::default()
            });
        }
    }
    for y in 0..n - 1 {
        for x in 0..n - 1 {
            let i = y * n + x;
            indices.push([i as u32, (i + 1) as u32, (i + n) as u32]);
            indices.push([(i + 1) as u32, (i + n + 1) as u32, (i + n) as u32]);
        }
    }

    let mesh = Mesh { vertices, indices };
    let original_faces = mesh.num_faces();
    assert_eq!(original_faces, 32);

    let decimated = vx_vision::mesh_ops::decimate(&mesh, 16);
    assert!(
        decimated.num_faces() <= 16,
        "should decimate to <= 16 faces, got {}",
        decimated.num_faces()
    );
    assert!(decimated.num_faces() > 0, "should still have some faces");
}

// ── Visualization: Point Cloud Renderer ──

#[cfg(all(feature = "reconstruction", feature = "visualization"))]
#[test]
fn point_cloud_renderer_produces_image() {
    use vx_vision::render_context::{Camera, RenderTarget};
    use vx_vision::renderers::point_cloud_renderer::PointCloudRenderer;
    use vx_vision::types_3d::{Point3D, PointCloud};

    let ctx = Context::new().unwrap();
    let renderer = PointCloudRenderer::new(&ctx).unwrap();

    // Create a few colored points in front of the camera
    let cloud = PointCloud {
        points: vec![
            Point3D {
                position: [0.0, 0.0, 0.0],
                color: [255, 0, 0, 255],
                normal: [0.0; 3],
            },
            Point3D {
                position: [0.5, 0.0, 0.0],
                color: [0, 255, 0, 255],
                normal: [0.0; 3],
            },
            Point3D {
                position: [0.0, 0.5, 0.0],
                color: [0, 0, 255, 255],
                normal: [0.0; 3],
            },
        ],
    };

    let target = RenderTarget::new(&ctx, 64, 64).unwrap();
    let camera = Camera::default();

    renderer
        .render(&ctx, &cloud, &camera, &target, 10.0)
        .unwrap();

    let result = target.read_rgba8();
    // Should have some non-black pixels
    let nonblack = result
        .chunks(4)
        .filter(|px| px[0] > 0 || px[1] > 0 || px[2] > 0)
        .count();
    assert!(nonblack > 0, "renderer should produce some colored pixels");
}

// ── Visualization: Mesh Renderer ──

#[cfg(all(feature = "reconstruction", feature = "visualization"))]
#[test]
fn mesh_renderer_produces_image() {
    use vx_vision::render_context::{Camera, RenderTarget};
    use vx_vision::renderers::mesh_renderer::MeshRenderer;
    use vx_vision::types_3d::{Mesh, Vertex3D};

    let ctx = Context::new().unwrap();
    let renderer = MeshRenderer::new(&ctx).unwrap();

    // Create a simple triangle facing the camera
    let mesh = Mesh {
        vertices: vec![
            Vertex3D {
                position: [-0.5, -0.5, 0.0],
                normal: [0.0, 0.0, 1.0],
                uv: [0.0; 2],
            },
            Vertex3D {
                position: [0.5, -0.5, 0.0],
                normal: [0.0, 0.0, 1.0],
                uv: [0.0; 2],
            },
            Vertex3D {
                position: [0.0, 0.5, 0.0],
                normal: [0.0, 0.0, 1.0],
                uv: [0.0; 2],
            },
        ],
        indices: vec![[0, 1, 2]],
    };

    let target = RenderTarget::new(&ctx, 64, 64).unwrap();
    let camera = Camera::default();

    renderer.render(&ctx, &mesh, &camera, &target).unwrap();

    let result = target.read_rgba8();
    // Background is dark (0.1, 0.1, 0.15), triangle should be lit
    let lit_pixels = result
        .chunks(4)
        .filter(|px| px[0] > 50 || px[1] > 50 || px[2] > 50)
        .count();
    assert!(
        lit_pixels > 0,
        "mesh renderer should produce some lit pixels"
    );
}

// ── 3D Reconstruction: Triangulation ──

#[cfg(feature = "reconstruction")]
#[test]
fn triangulate_two_view() {
    use vx_vision::kernels::triangulate::{Match2D, Triangulator};
    use vx_vision::types_3d::{CameraExtrinsics, CameraIntrinsics};

    let ctx = Context::new().unwrap();
    let triangulator = Triangulator::new(&ctx).unwrap();

    let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);

    // Camera 1 at origin looking along +Z
    // Camera 2 translated -1m along X in world (so t_world_to_cam = [-1, 0, 0])
    let pose1 = CameraExtrinsics::identity();
    let pose2 = CameraExtrinsics::new(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0], // world-to-camera: cam is at (1,0,0), so t = -(1,0,0)
    );

    // A 3D point at (0.5, 0.0, 5.0) in world space:
    // In camera 1: p_cam1 = (0.5, 0.0, 5.0) → u1 = 500*0.5/5 + 320 = 370, v1 = 240
    // In camera 2: p_cam2 = (0.5-1, 0.0, 5.0) = (-0.5, 0, 5) → u2 = 500*(-0.5)/5 + 320 = 270, v2 = 240
    let matches = vec![Match2D {
        u1: 370.0,
        v1: 240.0,
        u2: 270.0,
        v2: 240.0,
    }];

    let cloud = triangulator
        .triangulate(&ctx, &matches, &intrinsics, &intrinsics, &pose1, &pose2)
        .unwrap();

    assert!(
        cloud.len() >= 1,
        "should triangulate at least one point, got {}",
        cloud.len()
    );
    if cloud.len() > 0 {
        let p = cloud.points[0].position;
        // The triangulated point should be approximately at (0.5, 0, 5)
        assert!(
            p[2] > 1.0,
            "Z should be positive (in front of camera), got {}",
            p[2]
        );
    }
}
