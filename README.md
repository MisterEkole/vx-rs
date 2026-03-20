# VX

A computer vision library in Rust that talks directly to Apple Silicon GPUs through Metal Shading Language.

## Why

OpenCV and similar libraries treat the GPU as a separate device — data gets copied from CPU memory to GPU memory and back, over and over. On Apple Silicon this is wasteful because the CPU and GPU share the same physical memory (Unified Memory Architecture). VX skips the copies entirely.

The library uses Rust bindings to Metal via `objc2-metal`, giving us type-safe GPU access with Rust's ownership model enforcing buffer safety at compile time. Metal Shading Language (MSL) kernels run the actual pixel-level computation on the GPU, while Rust handles orchestration, memory management, and a clean public API.

The result: real-time classical vision algorithms with zero-copy memory, no C++ interop layer, and no Xcode project required.

## Architecture

VX is a three-layer stack:

```
┌─────────────────────────────────────────────┐
│  Application Layer          (your code)     │
│  Context::new() → detect() → corners        │
├─────────────────────────────────────────────┤
│  Kernel Layer               (vx-vision)     │
│  28+ GPU kernels across 6 categories        │
├─────────────────────────────────────────────┤
│  Memory Layer               (vx-core)       │
│  UnifiedBuffer<T> · GpuGuard<T> · Device    │
└─────────────────────────────────────────────┘
         ↕ zero-copy on Apple Silicon UMA
┌─────────────────────────────────────────────┐
│  Metal GPU                                  │
│  .metal shaders (compiled at build time)    │
└─────────────────────────────────────────────┘
```

**Naming convention:** In this codebase, *shaders* refer to the MSL functions that run on the GPU (`shaders/*.metal`), and *kernels* refer to the Rust bindings that orchestrate them (`src/kernels/*.rs`).

**Memory Layer** (`vx-core`) manages shared GPU/CPU buffers. `UnifiedBuffer<T>` wraps Metal buffers with type safety, and `GpuGuard<T>` prevents CPU access while a buffer is in-flight on the GPU.

**Kernel Layer** (`vx-vision`) contains Rust bindings for each MSL shader. Each kernel is a struct holding a compiled pipeline — constructed once, dispatched cheaply per frame. The `Context` and `Texture` wrappers hide all Metal internals so users never import `objc2-metal`.

**Application Layer** is your code. The API looks like this:

```rust
use vx_vision::Context;
use vx_vision::kernels::fast::{FastDetector, FastDetectConfig};
use vx_vision::kernels::harris::{HarrisScorer, HarrisConfig};

let ctx = Context::new()?;
let fast = FastDetector::new(&ctx)?;
let harris = HarrisScorer::new(&ctx)?;

let texture = ctx.texture_gray8(img.as_raw(), w, h)?;
let corners = fast.detect(&ctx, &texture, &FastDetectConfig::default())?;
let scored = harris.compute(&ctx, &texture, &corners.corners, &HarrisConfig::default())?;
```

No `unsafe` in user code. No Metal imports. No GPU boilerplate.

## Available Kernels

| Category | Kernels |
|---|---|
| **Feature Detection** | FAST-9, Harris, ORB descriptors, DoG/SIFT-like pipeline |
| **Image Processing** | Gaussian blur, bilateral filter, Sobel, Canny edge, morphology (erode/dilate), threshold (binary/Otsu), histogram (compute/equalize), color conversion |
| **Geometry** | Resize (bilinear), image pyramid, warp (affine/perspective), homography, lens undistortion |
| **Analysis** | Template matching (NCC), Hough lines, integral image, distance transform (JFA), connected components |
| **Motion & Stereo** | KLT optical flow, dense flow, stereo matching, brute-force descriptor matching |
| **Utilities** | Non-maximum suppression, texture pool, pipeline batching |

## Building

Requires macOS with Xcode command line tools (`xcode-select --install`).

```
cargo build
cargo test
cargo run --example fast_demo -- path/to/image.png
```

The build script automatically compiles all `.metal` shaders into a single metallib and embeds it in the binary.

See [docs/getting-started.md](docs/getting-started.md) for a detailed setup guide.

## Documentation

- [Getting Started](docs/getting-started.md) — setup, first project, core concepts
- [API Guide](docs/api-guide.md) — kernel-by-kernel reference with usage patterns
- [Performance](docs/performance.md) — TexturePool, Pipeline batching, optimization tips

## License

MIT
