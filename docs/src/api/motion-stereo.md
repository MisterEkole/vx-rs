# Motion & Stereo

## KLT Optical Flow

Sparse Kanade-Lucas-Tomasi tracker. Tracks keypoints across two frames using iterative Lucas-Kanade with image pyramids.

```rust
use vx_vision::kernels::klt::{KltTracker, KltConfig};

let klt = KltTracker::new(&ctx)?;
let mut cfg = KltConfig::default();
cfg.max_iterations = 30;
cfg.win_radius = 7;
cfg.max_level = 3;

let tracked = klt.track(&ctx, &prev_frame, &curr_frame, &keypoints, &cfg)?;
// tracked: Vec<KltResult> with new positions, status, and error
```

**Config:** `max_iterations`, `epsilon` (convergence threshold), `win_radius`, `max_level` (pyramid levels), `min_eigenvalue`.

## Dense Optical Flow

Horn-Schunck per-pixel flow estimation using iterative Jacobi relaxation.

```rust
use vx_vision::kernels::dense_flow::{DenseFlow, DenseFlowConfig};

let flow = DenseFlow::new(&ctx)?;
let mut cfg = DenseFlowConfig::default();
cfg.alpha = 0.012;
cfg.iterations = 50;

let result = flow.compute(&ctx, &frame0, &frame1, &cfg)?;
// result.flow_u: R32Float texture (horizontal displacement)
// result.flow_v: R32Float texture (vertical displacement)
```

Supports pipeline encoding via `flow.encode()`.

## Stereo Matching

Matches ORB features between rectified stereo image pairs using Hamming distance, epipolar constraints, and disparity bounds. Triangulates 3D positions from disparities.

```rust
use vx_vision::kernels::stereomatch::{StereoMatcher, StereoConfig};

let stereo = StereoMatcher::new(&ctx)?;
let mut cfg = StereoConfig::default();
cfg.max_disparity = 64.0;
cfg.baseline = 0.12;     // meters between cameras
cfg.fx = 500.0;          // focal length in pixels

let result = stereo.run(
    &ctx,
    &left_kpts, &right_kpts,
    &left_descs, &right_descs,
    &cfg,
)?;

for m in &result.matches {
    println!("3D: ({:.2}, {:.2}, {:.2})", m.point_3d[0], m.point_3d[1], m.point_3d[2]);
}
```

**Config:** `max_epipolar`, `min_disparity`, `max_disparity`, `max_hamming`, `ratio_thresh`, `fx`, `fy`, `cx`, `cy`, `baseline`.

## Brute-Force Descriptor Matching

Matches ORB binary descriptors using Hamming distance with Lowe's ratio test.

```rust
use vx_vision::kernels::matcher::{BruteMatcher, MatchConfig};

let matcher = BruteMatcher::new(&ctx)?;
let mut cfg = MatchConfig::default();
cfg.max_hamming = 64;
cfg.ratio_thresh = 0.75;

let matches = matcher.match_descriptors(&ctx, &query_desc, &train_desc, &cfg)?;
for m in &matches {
    println!("query[{}] → train[{}] dist={}", m.query_idx, m.train_idx, m.distance);
}
```

Descriptors are flat `&[u32]` arrays where every 8 consecutive values form one 256-bit ORB descriptor.
