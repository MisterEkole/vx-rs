# Analysis

## Template Matching

Normalized cross-correlation (NCC). Finds the best match location for a small template within a larger image.

```rust
use vx_vision::kernels::template_match::TemplateMatcher;

let tm = TemplateMatcher::new(&ctx)?;
let result = tm.match_template(&ctx, &image, &template)?;
println!("Best at ({}, {}) score={:.4}", result.best_x, result.best_y, result.best_score);
```

The template must have non-zero variance (not a uniform color) for NCC to produce meaningful results.

## Hough Line Detection

Detects lines via the Hough transform. Works best on binary edge images (e.g., output of Canny).

```rust
use vx_vision::kernels::hough::{HoughLines, HoughConfig};

let hough = HoughLines::new(&ctx)?;
let mut cfg = HoughConfig::default();
cfg.vote_threshold = 50;
cfg.max_lines = 100;

let lines = hough.detect(&ctx, &edge_image, &cfg)?;
for line in &lines {
    println!("rho={:.1} theta={:.1}° votes={}", line.rho, line.theta.to_degrees(), line.votes);
}
```

**Config:** `n_theta`, `edge_threshold`, `vote_threshold`, `max_lines`, `nms_radius`.

Not pipeline-encodable — requires CPU readback of the accumulator between voting and peak-finding.

## Integral Image

Summed area table for O(1) region-sum queries.

```rust
use vx_vision::kernels::integral::IntegralImage;

let integral = IntegralImage::new(&ctx)?;
let sat = integral.compute(&ctx, &input)?;
// sat: R32Float texture
```

Supports pipeline encoding via `integral.encode()`. Used internally by adaptive thresholding.

## Distance Transform

Euclidean distance from each pixel to the nearest seed pixel, computed via Jump Flooding Algorithm (JFA).

```rust
use vx_vision::kernels::distance::{DistanceTransform, DistanceConfig};

let dt = DistanceTransform::new(&ctx)?;
let mut cfg = DistanceConfig::default();
cfg.threshold = 0.5;

let distances = dt.compute(&ctx, &binary_input, &cfg)?;
// distances: R32Float texture with per-pixel Euclidean distance
```

## Connected Components

Labels connected regions in a binary image using iterative min-label propagation.

```rust
use vx_vision::kernels::connected::{ConnectedComponents, CCLConfig};

let ccl = ConnectedComponents::new(&ctx)?;
let mut cfg = CCLConfig::default();
cfg.threshold = 0.5;

let result = ccl.label(&ctx, &binary_input, &cfg)?;
println!("{} components in {} iterations", result.n_components, result.iterations);
// result.labels: R32Float texture with integer label per pixel
```

Not pipeline-encodable — iterative convergence requires CPU readback between passes.
