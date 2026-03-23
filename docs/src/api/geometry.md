# Geometry & Transforms

## Image Pyramid

Builds successive half-resolution levels in a single GPU submission.

```rust
use vx_vision::kernels::pyramid::PyramidBuilder;

let pyr = PyramidBuilder::new(&ctx)?;
let levels = pyr.build(&ctx, &input, 4)?;
// levels[0] = half, levels[1] = quarter, levels[2] = eighth

let half = pyr.downsample(&ctx, &input)?;  // single level
```

## Resize

Bilinear interpolation resize to arbitrary dimensions.

```rust
use vx_vision::kernels::resize::ImageResize;

let resizer = ImageResize::new(&ctx)?;
let output = resizer.apply(&ctx, &input, new_w, new_h)?;
```

## Warp

Affine and perspective warping.

```rust
use vx_vision::kernels::warp::ImageWarp;

let warp = ImageWarp::new(&ctx)?;

// Affine: 2x3 matrix as [f32; 6]
let output = ctx.texture_output_gray8(out_w, out_h)?;
warp.affine(&ctx, &input, &output, &matrix_2x3)?;

// Perspective: 3x3 matrix as [f32; 9]
warp.perspective(&ctx, &input, &output, &matrix_3x3)?;
```

Both support pipeline encoding via `encode_affine()` and `encode_perspective()`.

## Lens Undistortion

Corrects radial and tangential lens distortion using camera intrinsics.

```rust
use vx_vision::kernels::undistort::Undistorter;

let undistort = Undistorter::new(&ctx)?;
let output = undistort.apply(&ctx, &input, &camera_params)?;
```

## Homography Estimation

RANSAC-based homography from point correspondences. GPU-accelerated scoring with CPU-side model selection.

```rust
use vx_vision::kernels::homography::{HomographyEstimator, RansacConfig};

let estimator = HomographyEstimator::new(&ctx)?;
let mut cfg = RansacConfig::default();
cfg.max_iterations = 1000;
cfg.inlier_threshold = 3.0;

let result = estimator.estimate(&ctx, &point_pairs, &cfg)?;
// result.homography: [f32; 9]
// result.n_inliers: u32
// result.inlier_mask: Vec<bool>
```

Not pipeline-encodable — RANSAC iterates with CPU readback between GPU scoring passes.
