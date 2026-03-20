# API Guide

This guide covers every kernel in VX with usage patterns and configuration options.

## Feature Detection

### FAST-9 Corner Detector

Detects corners using the FAST-9 algorithm (Features from Accelerated Segment Test). Tests 16 pixels on a Bresenham circle of radius 3 — a pixel is a corner if 9 contiguous pixels are all brighter or darker than the center by a threshold.

```rust
use vx_vision::kernels::fast::{FastDetector, FastDetectConfig};

let fast = FastDetector::new(&ctx)?;
let result = fast.detect(&ctx, &input, &FastDetectConfig {
    threshold: 20,       // intensity difference threshold (i32)
    max_corners: 2048,   // maximum corners to return (u32)
})?;

for corner in &result.corners {
    println!("Corner at ({}, {}) score={}", corner.position[0], corner.position[1], corner.response);
}
```

### Harris Corner Response

Computes the Harris corner response score for detected keypoints. Use after FAST to rank corners by quality.

```rust
use vx_vision::kernels::harris::{HarrisScorer, HarrisConfig};

let harris = HarrisScorer::new(&ctx)?;
let scored = harris.compute(&ctx, &input, &corners, &HarrisConfig {
    k: 0.04,             // Harris sensitivity parameter
    patch_radius: 3,     // neighborhood radius around each keypoint
})?;
```

### ORB Descriptors

Computes ORB (Oriented FAST and Rotated BRIEF) binary descriptors for keypoints. Returns 256-bit descriptors for feature matching.

```rust
use vx_vision::kernels::orb::{OrbDescriptor, OrbConfig};

let orb = OrbDescriptor::new(&ctx)?;
let descriptors = orb.compute(&ctx, &input, &keypoints, &OrbConfig {
    patch_radius: 15,    // radius of the descriptor patch
})?;
```

### DoG Keypoint Detector

Difference-of-Gaussians detector for scale-space extrema (used in SIFT-like pipelines).

```rust
use vx_vision::kernels::dog::{DoGDetector, DoGConfig};

let dog = DoGDetector::new(&ctx)?;
let keypoints = dog.detect(&ctx, &input, &DoGConfig {
    n_levels: 5,
    base_sigma: 1.6,
    scale_factor: 1.2,
    contrast_threshold: 0.04,
    max_keypoints: 2048,
})?;
```

### SIFT-like Pipeline

Full SIFT-like pipeline: multi-octave Gaussian pyramid, DoG subtraction, extrema detection, orientation assignment, 128-dimensional descriptor computation.

```rust
use vx_vision::kernels::sift::{SiftPipeline, SiftConfig};

let sift = SiftPipeline::new(&ctx)?;
let features = sift.detect_and_describe(&ctx, &input, &SiftConfig::default())?;

for feat in &features {
    println!("Feature at ({}, {}) scale={} orientation={}",
        feat.x, feat.y, feat.scale, feat.orientation);
    // feat.descriptor is [f32; 128]
}

// Match features between two images
let matches = SiftPipeline::match_features(&features_a, &features_b, 0.75);
```

`SiftConfig` fields: `n_octaves`, `n_levels`, `base_sigma`, `contrast_threshold`, `edge_threshold`, `max_keypoints`, `descriptor_radius`.

## Image Processing

### Gaussian Blur

Separable Gaussian blur (horizontal + vertical passes).

```rust
use vx_vision::kernels::gaussian::{GaussianBlur, GaussianConfig};

let blur = GaussianBlur::new(&ctx)?;
let output = blur.apply(&ctx, &input, &GaussianConfig {
    sigma: 2.0,
    radius: 5,
})?;
```

### Bilateral Filter

Edge-preserving blur. Smooths flat regions while keeping edges sharp.

```rust
use vx_vision::kernels::bilateral::{BilateralFilter, BilateralConfig};

let bilateral = BilateralFilter::new(&ctx)?;
let output = bilateral.apply(&ctx, &input, &BilateralConfig {
    radius: 5,
    sigma_spatial: 3.0,
    sigma_range: 0.1,
})?;
```

### Sobel Edge Detection

Computes gradient magnitude and direction using Sobel operators.

```rust
use vx_vision::kernels::sobel::SobelFilter;

let sobel = SobelFilter::new(&ctx)?;
let (magnitude, direction) = sobel.compute(&ctx, &input)?;
// magnitude: R32Float texture with gradient magnitude
// direction: R32Float texture with gradient angle in radians

// Or get just the gradient images (Gx, Gy) without computing magnitude/direction:
let (gx, gy) = sobel.gradients_only(&ctx, &input)?;
```

### Canny Edge Detection

Multi-stage edge detector: Gaussian blur, Sobel gradients, non-maximum suppression, hysteresis thresholding.

```rust
use vx_vision::kernels::canny::{CannyDetector, CannyConfig};

let canny = CannyDetector::new(&ctx)?;
let edges = canny.detect(&ctx, &input, &CannyConfig {
    low_threshold: 0.1,
    high_threshold: 0.3,
    blur_sigma: 1.4,
    blur_radius: 3,
})?;
```

### Morphology (Erode / Dilate / Open / Close)

Binary morphological operations with a configurable rectangular structuring element.

```rust
use vx_vision::kernels::morphology::{Morphology, MorphConfig};

let morph = Morphology::new(&ctx)?;
let config = MorphConfig { radius_x: 1, radius_y: 1 }; // 3x3 kernel

let eroded = morph.erode(&ctx, &input, &config)?;
let dilated = morph.dilate(&ctx, &input, &config)?;

// Opening (erode then dilate) removes small noise
let opened = morph.open(&ctx, &input, &config)?;

// Closing (dilate then erode) fills small holes
let closed = morph.close(&ctx, &input, &config)?;
```

### Threshold

Binary thresholding, adaptive thresholding, and Otsu's automatic threshold selection.

```rust
use vx_vision::kernels::threshold::Threshold;

let thresh = Threshold::new(&ctx)?;

// Fixed binary threshold
let binary = thresh.binary(&ctx, &input, &output, 128)?;

// Otsu's method (automatic threshold, returns the computed threshold value)
let otsu_value = thresh.otsu(&ctx, &input, &output)?;

// Adaptive thresholding
use vx_vision::kernels::threshold::AdaptiveThresholdConfig;
let adaptive = thresh.adaptive(&ctx, &input, &output, &AdaptiveThresholdConfig {
    radius: 7,
    c: 5.0,
    invert: false,
})?;
```

### Histogram

Compute and equalize image histograms.

```rust
use vx_vision::kernels::histogram::Histogram;

let hist = Histogram::new(&ctx)?;
let bins = hist.compute(&ctx, &input)?;        // [u32; 256]
let equalized = hist.equalize(&ctx, &input)?;  // contrast-enhanced output
```

### Color Conversion

Convert between color spaces.

```rust
use vx_vision::kernels::color::ColorConvert;

let color = ColorConvert::new(&ctx)?;
let gray = color.rgba_to_gray(&ctx, &rgba_input)?;
let rgba = color.gray_to_rgba(&ctx, &gray_input)?;
let hsv = color.rgba_to_hsv(&ctx, &rgba_input)?;
let back = color.hsv_to_rgba(&ctx, &hsv)?;
```

## Geometry

### Resize

Bilinear interpolation resize.

```rust
use vx_vision::kernels::resize::ImageResize;

let resizer = ImageResize::new(&ctx)?;
let output = resizer.apply(&ctx, &input, new_width, new_height)?;
```

### Image Pyramid

Build a Gaussian image pyramid (successive half-resolution levels).

```rust
use vx_vision::kernels::pyramid::PyramidBuilder;

let pyr = PyramidBuilder::new(&ctx)?;
// n_levels includes the input; returns n_levels-1 downsampled textures
let levels = pyr.build(&ctx, &input, 4)?;
// levels[0] = half size, levels[1] = quarter size, levels[2] = eighth

// Single downsample step
let half = pyr.downsample(&ctx, &input)?;
```

### Warp

Affine and perspective image warping.

```rust
use vx_vision::kernels::warp::ImageWarp;

let warp = ImageWarp::new(&ctx)?;

// Affine warp (2x3 matrix)
let warped = warp.affine(&ctx, &input, &affine_matrix, out_w, out_h)?;

// Perspective warp (3x3 matrix)
let warped = warp.perspective(&ctx, &input, &perspective_matrix, out_w, out_h)?;
```

### Lens Undistortion

Remove lens distortion using camera intrinsics and distortion coefficients.

```rust
use vx_vision::kernels::undistort::Undistorter;

let undistort = Undistorter::new(&ctx)?;
let corrected = undistort.apply(&ctx, &input, &camera_params)?;
```

### Homography Estimation

Estimate a homography (3x3 perspective transform) between two sets of point correspondences using RANSAC.

```rust
use vx_vision::kernels::homography::{HomographyEstimator, RansacConfig};

let estimator = HomographyEstimator::new(&ctx)?;
let result = estimator.estimate(&ctx, &src_points, &dst_points, &RansacConfig {
    max_iterations: 1000,
    inlier_threshold: 3.0,
    min_inliers: 10,
})?;
```

## Analysis

### Template Matching

Normalized cross-correlation (NCC) template matching. Finds the best match location for a template within a larger image.

```rust
use vx_vision::kernels::template_match::TemplateMatcher;

let tm = TemplateMatcher::new(&ctx)?;
let result = tm.match_template(&ctx, &image, &template)?;
println!("Best match at ({}, {}) score={}", result.best_x, result.best_y, result.best_score);
```

Note: The template must have non-zero variance (not a uniform color) for NCC to produce meaningful results.

### Hough Line Detection

Detect lines using the Hough transform. Works best on binary edge images (output of Canny).

```rust
use vx_vision::kernels::hough::{HoughLines, HoughConfig};

let hough = HoughLines::new(&ctx)?;
let lines = hough.detect(&ctx, &edge_image, &HoughConfig {
    n_theta: 180,
    edge_threshold: 128,
    vote_threshold: 50,
    max_lines: 100,
    nms_radius: 5,
})?;
for line in &lines {
    println!("Line: rho={}, theta={}", line.rho, line.theta);
}
```

### Integral Image

Compute the integral image (summed area table) for fast region-sum queries.

```rust
use vx_vision::kernels::integral::IntegralImage;

let integral = IntegralImage::new(&ctx)?;
let sat = integral.compute(&ctx, &input)?;
```

### Distance Transform

Compute the distance transform using the Jump Flooding Algorithm (JFA). Returns the Euclidean distance from each pixel to the nearest non-zero pixel.

```rust
use vx_vision::kernels::distance::{DistanceTransform, DistanceConfig};

let dt = DistanceTransform::new(&ctx)?;
let distances = dt.compute(&ctx, &binary_input, &DistanceConfig {
    threshold: 0.5,
})?;
```

### Connected Components

Label connected components in a binary image using iterative min-label propagation.

```rust
use vx_vision::kernels::connected::{ConnectedComponents, CCLConfig};

let ccl = ConnectedComponents::new(&ctx)?;
let labels = ccl.label(&ctx, &binary_input, &CCLConfig {
    threshold: 0.5,
    max_iterations: 100,
})?;
```

## Motion & Stereo

### KLT Optical Flow

Kanade-Lucas-Tomasi sparse optical flow tracking.

```rust
use vx_vision::kernels::klt::{KltTracker, KltConfig};

let klt = KltTracker::new(&ctx)?;
let tracked = klt.track(&ctx, &prev_frame, &curr_frame, &keypoints, &KltConfig {
    max_iterations: 30,
    epsilon: 0.01,
    win_radius: 7,
    max_level: 3,
    min_eigenvalue: 1e-4,
})?;
```

### Dense Optical Flow

Dense (per-pixel) optical flow estimation.

```rust
use vx_vision::kernels::dense_flow::{DenseFlow, DenseFlowConfig};

let flow = DenseFlow::new(&ctx)?;
let flow_field = flow.compute(&ctx, &prev_frame, &curr_frame, &DenseFlowConfig {
    alpha: 0.012,
    iterations: 50,
})?;
```

### Stereo Matching

Disparity estimation from stereo image pairs.

```rust
use vx_vision::kernels::stereomatch::{StereoMatcher, StereoConfig};

let stereo = StereoMatcher::new(&ctx)?;
let disparity = stereo.run(&ctx, &left, &right, &StereoConfig {
    max_disparity: 64,
    min_disparity: 0,
    max_hamming: 64,
    ratio_thresh: 0.8,
    max_epipolar: 2.0,
    fx: 500.0, fy: 500.0,
    cx: 320.0, cy: 240.0,
    baseline: 0.12,
})?;
```

### Brute-Force Descriptor Matching

Match binary descriptors (e.g., ORB) between two sets using Hamming distance with ratio test.

```rust
use vx_vision::kernels::matcher::{BruteMatcher, MatchConfig};

let matcher = BruteMatcher::new(&ctx)?;
let matches = matcher.match_descriptors(&ctx, &desc_a, &desc_b, &MatchConfig {
    max_hamming: 64,
    ratio_thresh: 0.75,
})?;
```

## Utilities

### Non-Maximum Suppression

Suppress non-maximal detections within a neighborhood.

```rust
use vx_vision::kernels::nms::{NmsSuppressor, NmsConfig};

let nms = NmsSuppressor::new(&ctx)?;
let suppressed = nms.run(&ctx, &corners, &NmsConfig {
    min_distance: 10.0,
})?;
```

### TexturePool

Recycle GPU textures by `(width, height, format)` to avoid repeated allocations. See [Performance Guide](performance.md) for details.

```rust
use vx_vision::TexturePool;

let mut pool = TexturePool::new();
let tex = pool.acquire_gray8(&ctx, 640, 480)?;
// ... use tex ...
pool.release(tex);
```

### Pipeline

Batch multiple kernel dispatches into a single Metal command buffer. See [Performance Guide](performance.md) for details.

```rust
use vx_vision::Pipeline;

let pipe = Pipeline::begin(&ctx)?;
blur.encode(pipe.cmd_buf(), &input, &temp)?;
sobel.encode(pipe.cmd_buf(), &temp, &output)?;
pipe.commit_and_wait();
```

## Camera Integration

For real-time camera pipelines using AVFoundation, wrap existing Metal textures without copying:

```rust
use vx_vision::{Texture, TextureFormat};

// In your AVFoundation capture callback:
let tex = Texture::from_metal_texture(metal_texture, width, height, TextureFormat::RGBA8Unorm);
// Process with any kernel — zero copy on UMA
```
