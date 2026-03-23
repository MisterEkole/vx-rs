# Image Processing

## Gaussian Blur

Separable two-pass blur (horizontal then vertical).

```rust
use vx_vision::kernels::gaussian::{GaussianBlur, GaussianConfig};

let blur = GaussianBlur::new(&ctx)?;
let output = ctx.texture_output_gray8(w, h)?;
let mut cfg = GaussianConfig::default();  // sigma: 1.0, radius: 3
cfg.sigma = 2.0;

blur.apply(&ctx, &input, &output, &cfg)?;
```

**Config:** `sigma` (standard deviation), `radius` (kernel half-width, full kernel = 2*radius + 1).

Pipeline encoding returns a `GaussianEncodedState` that holds the intermediate texture:

```rust
let state = blur.encode(&ctx, cmd_buf, &input, &output, &cfg)?;
// state must outlive the command buffer
```

## Bilateral Filter

Edge-preserving smoothing. Smooths flat regions while keeping edges sharp.

```rust
use vx_vision::kernels::bilateral::{BilateralFilter, BilateralConfig};

let bilateral = BilateralFilter::new(&ctx)?;
let output = ctx.texture_output_gray8(w, h)?;
bilateral.apply(&ctx, &input, &output, &BilateralConfig::new(5, 10.0, 0.1))?;
```

**Config:** `radius`, `sigma_spatial`, `sigma_range`. Larger `sigma_range` allows more intensity variation.

## Sobel Edge Detection

Computes gradient magnitude and direction.

```rust
use vx_vision::kernels::sobel::SobelFilter;

let sobel = SobelFilter::new(&ctx)?;
let result = sobel.compute(&ctx, &input)?;
// result.magnitude: R32Float texture
// result.direction: R32Float texture (radians)
// result.grad_x, result.grad_y: R32Float gradient components
```

## Canny Edge Detection

Multi-stage: Gaussian blur → Sobel → non-maximum suppression → hysteresis thresholding.

```rust
use vx_vision::kernels::canny::{CannyDetector, CannyConfig};

let canny = CannyDetector::new(&ctx)?;
let mut cfg = CannyConfig::default();
cfg.low_threshold = 0.04;
cfg.high_threshold = 0.12;

let edges = canny.detect(&ctx, &input, &cfg)?;
// edges: R32Float texture (1.0 = edge, 0.0 = non-edge)
```

**Config:** `low_threshold`, `high_threshold` (hysteresis), `blur_sigma`, `blur_radius`.

Supports pipeline encoding via `canny.encode()`.

## Morphology

Binary operations with a rectangular structuring element.

```rust
use vx_vision::kernels::morphology::{Morphology, MorphConfig};

let morph = Morphology::new(&ctx)?;
let cfg = MorphConfig::default();  // radius_x: 1, radius_y: 1 (3x3 kernel)
let output = ctx.texture_output_gray8(w, h)?;

morph.erode(&ctx, &input, &output, &cfg)?;
morph.dilate(&ctx, &input, &output, &cfg)?;
morph.open(&ctx, &input, &output, &cfg)?;    // erode then dilate
morph.close(&ctx, &input, &output, &cfg)?;   // dilate then erode
```

All four operations support pipeline encoding: `encode_erode`, `encode_dilate`, `encode_open`, `encode_close`.

## Threshold

Binary, adaptive, and automatic (Otsu) thresholding.

```rust
use vx_vision::kernels::threshold::{Threshold, AdaptiveThresholdConfig};

let thresh = Threshold::new(&ctx)?;
let output = ctx.texture_output_gray8(w, h)?;

// Fixed binary (normalized 0.0-1.0 threshold)
thresh.binary(&ctx, &input, &output, 0.5, false)?;

// Otsu's method (auto-selects threshold, returns it)
let value = thresh.otsu(&ctx, &input, &output)?;

// Adaptive (requires integral image)
let cfg = AdaptiveThresholdConfig::new(15, 0.03, false);
thresh.adaptive_auto(&ctx, &input, &output, &cfg)?;
```

Pipeline encoding: `encode_binary()`, `encode_adaptive()`.

## Histogram

Compute 256-bin histogram and equalize contrast.

```rust
use vx_vision::kernels::histogram::Histogram;

let hist = Histogram::new(&ctx)?;
let bins: [u32; 256] = hist.compute(&ctx, &input)?;
let output = ctx.texture_output_gray8(w, h)?;
hist.equalize(&ctx, &input, &output)?;
```

Not pipeline-encodable — requires CPU readback of bin counts.

## Color Conversion

Convert between RGBA, grayscale, and HSV.

```rust
use vx_vision::kernels::color::ColorConvert;

let color = ColorConvert::new(&ctx)?;
color.rgba_to_gray(&ctx, &rgba, &gray)?;
color.gray_to_rgba(&ctx, &gray, &rgba)?;
color.rgba_to_hsv(&ctx, &rgba, &hsv)?;
color.hsv_to_rgba(&ctx, &hsv, &rgba)?;
```

All four conversions support pipeline encoding.
