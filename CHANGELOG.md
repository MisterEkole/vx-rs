# Changelog

All notable changes to vx-rs will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.3.0] - 2026-03-30

### Added

#### 3D Reconstruction Pipeline
- **14 new Metal compute shaders** for 3D geometry processing
- `DepthToCloud` — GPU-accelerated depth map to point cloud conversion
- `NormalEstimation` — surface normal computation from point clouds
- `Triangulate` — point cloud triangulation to mesh
- `MarchingCubes` — isosurface extraction from volumetric data
- `TSDFIntegrate` / `TSDFRaycast` — truncated signed distance function for volumetric fusion
- `SGMStereo` — semi-global matching for dense stereo disparity
- `VoxelDownsample` — spatial downsampling of point clouds
- `OutlierFilter` — statistical outlier removal from point clouds

#### Depth Processing
- `DepthFilter` — bilateral/median filtering for depth maps
- `DepthInpaint` — hole-filling for sparse depth maps
- `DepthColorize` — depth-to-color visualization with configurable colormaps

#### Visualization
- `RenderContext` with offscreen `RenderTarget` (RGBA8 color + Depth32Float)
- `PointCloudRenderer` and `MeshRenderer` for 3D data visualization
- `PointCloudRender.metal` and `MeshRender.metal` shaders

#### Data Types (`types_3d`)
- `Point3D` — position, color, and normal vector
- `Vertex3D` — mesh vertex with position, normal, and UV coordinates
- `PointCloud` and `Mesh` container types
- `CameraIntrinsics` and `CameraExtrinsics` for camera models
- `BoundingBox3D`, `VoxelGrid`, `TSDFVolume`, and more

#### Export & Datasets
- PLY export for point clouds and meshes (`export` module)
- Dataset loaders for EuRoC, KITTI, and TUM RGB-D benchmarks (`datasets` module)
- `DatasetFrame` and `DatasetIterator` abstractions

#### Feature Flags
- `reconstruction` — 3D reconstruction kernels, export, mesh ops
- `visualization` — render targets and renderers
- `datasets` — benchmark dataset loaders
- `full` — enables all features

#### Other
- New examples: `depth_to_cloud_demo`, `point_cloud_processing_demo`, `tsdf_fusion_demo`
- Expanded benchmarks for all new kernels
- Design document (`docs/src/design.md`)
- API documentation for reconstruction and visualization modules

### Changed
- Refactored existing kernels to use consistent encode/commit patterns
- Improved texture format handling across all kernels
- Updated examples to use new API patterns

---

## [0.2.0] - 2025-12-01

### Added
- Published to crates.io as `vx-gpu` and `vx-vision`
- Renamed `vx-core` to `vx-gpu`
- NMS (non-maximum suppression) kernel and shader
- Gaussian blur kernel and shader
- Undistort kernel with R32Float and output texture support
- ORB descriptor and stereo matcher kernels
- KLT optical flow tracker kernel
- ORB + StereoMatch benchmark example
- Comprehensive kernel tests and documentation

---

## [0.1.0] - 2025-10-01

### Added
- Initial release
- Metal compute shader build system with automatic `.metal` discovery
- `UnifiedBuffer<T>` and `GpuGuard<T>` for zero-copy UMA memory
- Core kernels: FAST, Harris, Sobel, Canny, Threshold, Bilateral, Morphology, Histogram, Hough, Homography, Integral, Pyramid, Resize, Warp, Distance Transform, Template Match, Connected Components, Dense Optical Flow, DoG, SIFT, Color conversion
- `Context`, `Texture`, `Pipeline`, and `TexturePool` abstractions
- Examples: `fast_demo`, `edge_detection_demo`, `advanced_cv_demo`, and more
