# 3D Reconstruction

All kernels require the `reconstruction` feature flag: `cargo build --features reconstruction`

## SGM Stereo

Semi-Global Matching stereo produces dense disparity maps from rectified stereo pairs. Uses Census transform with 8-path cost aggregation and sub-pixel refinement.

```rust
use vx_vision::kernels::sgm::{SGMStereo, SGMStereoConfig};

let sgm = SGMStereo::new(&ctx)?;
let config = SGMStereoConfig::new(128); // 128 disparity levels
let output = ctx.texture_output_r32float(w, h)?;

sgm.compute_disparity(&ctx, &left, &right, &output, &config)?;
// output: R32Float texture with per-pixel disparity values
```

**Config:** `num_disparities` (search range), `p1` (penalty for ±1 disparity change), `p2` (penalty for larger changes), `census_radius_x`, `census_radius_y`.

## Depth Filter

Depth-aware bilateral filter and median filter for cleaning up noisy depth/disparity maps. Preserves depth edges while smoothing.

```rust
use vx_vision::kernels::depth_filter::{DepthFilter, DepthFilterConfig, DepthMedianConfig};

let filter = DepthFilter::new(&ctx)?;

// Bilateral: edge-preserving smoothing
let config = DepthFilterConfig::new(3, 5.0, 0.05);
filter.apply_bilateral(&ctx, &input, &output, &config)?;

// Median: salt-and-pepper noise removal
let med_config = DepthMedianConfig::default(); // 3×3
filter.apply_median(&ctx, &input, &output, &med_config)?;
```

Supports pipeline encoding via `filter.encode_bilateral()` and `filter.encode_median()`.

## Depth Inpaint

Fills holes in depth maps using iterative nearest-neighbor propagation (jump-flooding pattern).

```rust
use vx_vision::kernels::depth_inpaint::{DepthInpaint, DepthInpaintConfig};

let inpaint = DepthInpaint::new(&ctx)?;
let mut config = DepthInpaintConfig::default();
config.max_iterations = 6; // doubles search radius each iteration

inpaint.apply(&ctx, &input, &output, &config)?;
```

## Depth-to-Point-Cloud

GPU-accelerated unprojection of depth maps to 3D point clouds. One thread per pixel, with atomic compaction to skip invalid pixels.

```rust
use vx_vision::kernels::depth_to_cloud::{DepthToCloud, DepthToCloudConfig};
use vx_vision::types_3d::{CameraIntrinsics, DepthMap};

let d2c = DepthToCloud::new(&ctx)?;
let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
let depth_map = DepthMap::new(depth_texture, intrinsics, 0.1, 10.0)?;

let config = DepthToCloudConfig::new(0.1, 10.0);
let cloud = d2c.compute(&ctx, &depth_map, Some(&color_texture), &config)?;
// cloud: PointCloud with XYZ + optional RGB per point
```

**Config:** `min_depth`, `max_depth`, `depth_scale`, `max_points`.

## Depth Colorize

Maps depth values to RGBA colors using a colormap for visualization.

```rust
use vx_vision::kernels::depth_colorize::{DepthColorize, DepthColorizeConfig};

let colorize = DepthColorize::new(&ctx)?;
let config = DepthColorizeConfig::new(0.5, 5.0); // min/max depth range

colorize.apply(&ctx, &depth_r32, &rgba_output, &config)?;
```

Supports Turbo (default), Jet, and Inferno colormaps. Pipeline encoding via `colorize.encode()`.

## Normal Estimation

Estimates surface normals for point clouds. Two modes: organized (fast, from depth maps) and unorganized (brute-force k-NN PCA).

```rust
use vx_vision::kernels::normal_estimation::{NormalEstimator, NormalEstimatorConfig};

let estimator = NormalEstimator::new(&ctx)?;

// Organized: from depth map (fast — cross product of adjacent pixels)
estimator.compute_from_depth(&ctx, &depth_tex, &normal_out, fx, fy, cx, cy)?;

// Unorganized: from arbitrary point cloud
let mut config = NormalEstimatorConfig::default();
config.radius = 0.1;
let normals = estimator.compute(&ctx, &point_cloud, &config)?;
// normals: Vec<[f32; 3]>
```

## Outlier Filter

Statistical outlier removal. Computes mean distance to k-nearest neighbors per point, then rejects points beyond `mean + std_ratio × stddev`.

```rust
use vx_vision::kernels::outlier_filter::{OutlierFilter, OutlierFilterConfig};

let filter = OutlierFilter::new(&ctx)?;
let config = OutlierFilterConfig::default(); // k=10, std_ratio=2.0

let filtered = filter.filter(&ctx, &cloud, &config)?;
```

## Voxel Downsample

Reduces point cloud density by averaging points within each voxel cell. Uses GPU hash table with atomic accumulation.

```rust
use vx_vision::kernels::voxel_downsample::{VoxelDownsample, VoxelDownsampleConfig};

let ds = VoxelDownsample::new(&ctx)?;
let config = VoxelDownsampleConfig::new(0.05); // 5cm voxel size

let downsampled = ds.downsample(&ctx, &cloud, &config)?;
```

## TSDF Volume Fusion

Truncated Signed Distance Function — the core of real-time 3D reconstruction. Integrates sequential depth frames into a volumetric representation and raycasts synthetic views.

```rust
use vx_vision::kernels::tsdf::{TSDFVolume, TSDFConfig};
use vx_vision::types_3d::{CameraExtrinsics, CameraIntrinsics};

let mut config = TSDFConfig::default();
config.resolution = [256, 256, 256];
config.voxel_size = 0.005; // 5mm
let tsdf = TSDFVolume::new(&ctx, config)?;

// Integrate a depth frame
tsdf.integrate(&ctx, &depth_map, &camera_pose)?;

// Raycast synthetic view
let (depth_out, normal_out) = tsdf.raycast(&ctx, &pose, &intrinsics)?;

// Extract surface points
let cloud = tsdf.extract_cloud();
```

**Config:** `resolution`, `voxel_size`, `truncation_distance`, `max_weight`, `origin`.

## Marching Cubes

Extracts a triangle mesh from a TSDF volume at the zero-crossing surface. CPU-side with full 256-entry lookup table, reading directly from GPU shared memory (zero-copy UMA).

```rust
use vx_vision::kernels::marching_cubes::{MarchingCubes, MarchingCubesConfig};

let config = MarchingCubesConfig::default(); // iso_level = 0.0
let mut mesh = MarchingCubes::extract(tsdf.volume(), &config);

mesh.compute_normals();
mesh.weld_vertices(0.001);
```

## Triangulation

GPU-accelerated midpoint triangulation from two-view 2D-2D correspondences.

```rust
use vx_vision::kernels::triangulate::{Triangulator, Match2D};

let tri = Triangulator::new(&ctx)?;
let matches = vec![
    Match2D { u1: 320.0, v1: 240.0, u2: 280.0, v2: 240.0 },
];

let cloud = tri.triangulate(&ctx, &matches, &intrinsics1, &intrinsics2, &pose1, &pose2)?;
```

## Mesh Operations (CPU)

```rust
use vx_vision::mesh_ops;

// Decimate to target face count (edge-collapse)
let decimated = mesh_ops::decimate(&mesh, 5000);
```

Mesh types also provide:
- `mesh.compute_normals()` — per-vertex normals from face normals
- `mesh.weld_vertices(tolerance)` — merge duplicate vertices

## Export Formats

```rust
use vx_vision::export;

export::write_ply_ascii("cloud.ply", &point_cloud)?;
export::write_ply_binary("cloud.ply", &point_cloud)?;
export::write_obj("mesh.obj", &mesh)?;
export::write_mesh_ply("mesh.ply", &mesh)?;
```

## Core 3D Types

| Type | Description |
|------|-------------|
| `Point3D` | Position + color + normal |
| `Vertex3D` | Position + normal + UV |
| `PointCloud` | Collection of `Point3D` with `bounds()`, `len()`, `positions()` |
| `Mesh` | Indexed triangle mesh with `compute_normals()`, `weld_vertices()` |
| `DepthMap` | R32Float texture + intrinsics + depth range |
| `CameraIntrinsics` | Pinhole model: `fx, fy, cx, cy, width, height` |
| `CameraExtrinsics` | Rotation (3x3) + translation, with `transform_point()`, `inverse()`, `to_gpu_rows()` |
| `VoxelGrid` | TSDF + weights backed by `UnifiedBuffer`, with `voxel_to_world()`, `reset()` |
