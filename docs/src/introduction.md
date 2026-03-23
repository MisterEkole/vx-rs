# VX

GPU-accelerated computer vision for Rust on Apple Silicon.

VX talks directly to the Metal GPU through compute shaders, using Apple Silicon's Unified Memory Architecture to eliminate the CPU-GPU copy overhead that plagues traditional vision libraries.

## What it does

42 GPU kernels covering classical computer vision and 3D reconstruction: feature detection (FAST, Harris, ORB, SIFT), image processing (Gaussian, bilateral, Canny, morphology, thresholding), geometry (pyramids, warping, homography), motion (KLT tracking, dense flow), stereo matching, 3D reconstruction (SGM stereo, TSDF fusion, marching cubes, point cloud processing), visualization (mesh and point cloud renderers), and analysis (Hough lines, template matching, distance transforms, connected components).

## Why it exists

OpenCV and similar libraries treat the GPU as a separate device. Data gets copied from CPU memory to GPU memory and back, repeatedly. On Apple Silicon this is wasteful — the CPU and GPU share the same physical memory. VX skips the copies entirely.

The library uses Rust bindings to Metal via `objc2-metal`, giving type-safe GPU access with Rust's ownership model enforcing buffer safety at compile time. Metal Shading Language (MSL) kernels run the pixel-level computation on the GPU while Rust handles orchestration and the public API.

## Quick taste

```rust
use vx_vision::Context;
use vx_vision::kernels::fast::{FastDetector, FastDetectConfig};

let ctx = Context::new()?;
let texture = ctx.texture_gray8(&pixels, width, height)?;

let fast = FastDetector::new(&ctx)?;
let result = fast.detect(&ctx, &texture, &FastDetectConfig::default())?;

println!("Found {} corners", result.corners.len());
```

No `unsafe` in user code. No Metal imports. No GPU boilerplate.
