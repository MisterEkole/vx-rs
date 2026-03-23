# Feature Detection

## FAST-9 Corner Detector

Detects corners using the FAST-9 algorithm. Tests 16 pixels on a Bresenham circle — a pixel is a corner if 9 contiguous pixels are all brighter or darker than the center by a threshold.

```rust
use vx_vision::kernels::fast::{FastDetector, FastDetectConfig};

let fast = FastDetector::new(&ctx)?;
let mut cfg = FastDetectConfig::default();  // threshold: 20, max_corners: 2048
cfg.threshold = 30;

let result = fast.detect(&ctx, &input, &cfg)?;
for corner in &result.corners {
    println!("({}, {}) score={}", corner.position[0], corner.position[1], corner.response);
}
```

Also supports pipeline encoding via `fast.encode()`.

## Harris Corner Response

Computes the Harris response `R = det(M) - k * trace(M)^2` for each keypoint. Use after FAST to rank corners by quality.

```rust
use vx_vision::kernels::harris::{HarrisScorer, HarrisConfig};

let harris = HarrisScorer::new(&ctx)?;
let scored = harris.compute(&ctx, &input, &corners, &HarrisConfig::default())?;
// scored: Vec<CornerPoint> with updated response values
```

**Config:** `k` (sensitivity, default 0.04), `patch_radius` (neighborhood size, default 3).

## Non-Maximum Suppression

Filters keypoints so no two are within `min_distance` of each other. Keeps the highest-response point in each neighborhood.

```rust
use vx_vision::kernels::nms::{NmsSuppressor, NmsConfig};

let nms = NmsSuppressor::new(&ctx)?;
let filtered = nms.run(&ctx, &corners, &NmsConfig::default())?;
```

**Config:** `min_distance` (default 10.0 pixels).

## ORB Descriptors

Computes 256-bit binary descriptors for keypoints using oriented BRIEF test pairs.

```rust
use vx_vision::kernels::orb::{OrbDescriptor, OrbConfig};

let orb = OrbDescriptor::new(&ctx)?;
let result = orb.compute(&ctx, &input, &keypoints, &pattern, &OrbConfig::default())?;
// result.descriptors: Vec<ORBOutput> (256-bit descriptors as 8 x u32)
// result.orientations: Vec<f32>
```

The `pattern` is 1024 `i32` values (256 test pairs, each with 4 offsets: dx1, dy1, dx2, dy2).

## DoG Keypoint Detector

Difference-of-Gaussians scale-space extrema detection.

```rust
use vx_vision::kernels::dog::{DoGDetector, DoGConfig};
use vx_vision::kernels::gaussian::GaussianBlur;

let blur = GaussianBlur::new(&ctx)?;
let dog = DoGDetector::new(&ctx)?;
let mut cfg = DoGConfig::default();
cfg.n_levels = 5;

let keypoints = dog.detect(&ctx, &blur, &input, &cfg)?;
```

Each keypoint has `position`, `scale`, and `response`. Full pipelining isn't practical due to the iterative blur-subtract-extrema pattern, but `encode_subtract()` exposes the subtraction step for custom pipelines.

## SIFT Pipeline

Full SIFT-like pipeline: multi-octave pyramid, DoG detection, orientation assignment, 128-dimensional descriptors.

```rust
use vx_vision::kernels::sift::{SiftPipeline, SiftConfig};

let sift = SiftPipeline::new(&ctx)?;
let features = sift.detect_and_describe(&ctx, &input, &SiftConfig::default())?;

for f in &features {
    println!("({}, {}) scale={:.2} orient={:.2}", f.x, f.y, f.scale, f.orientation);
    // f.descriptor: [f32; 128]
}
```

Matching between two feature sets:

```rust
let matches = SiftPipeline::match_features(&features_a, &features_b, 0.75);
```

## Typical detection pipeline

A common pattern chains FAST → Harris → NMS → ORB:

```rust
let corners = fast.detect(&ctx, &texture, &fast_cfg)?;
let scored  = harris.compute(&ctx, &texture, &corners.corners, &harris_cfg)?;
let best    = nms.run(&ctx, &scored, &nms_cfg)?;
let descs   = orb.compute(&ctx, &texture, &best, &pattern, &orb_cfg)?;
```

For single-submission batching, each of these kernels provides an `encode()` method that writes into a shared command buffer via `Pipeline`.
