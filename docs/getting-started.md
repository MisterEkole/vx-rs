# Getting Started with VX

## Requirements

- macOS on Apple Silicon (M1/M2/M3/M4) or a Mac with a Metal-compatible GPU
- Rust (stable toolchain)
- Xcode command line tools: `xcode-select --install`

## Installation

Add VX to your `Cargo.toml`:

```toml
[dependencies]
vx-vision = { path = "../vx-rs/vx-vision" }
```

Or if you've published to a registry:

```toml
[dependencies]
vx-vision = "0.1"
```

## Core Concepts

### Context

`Context` is the entry point. It initializes the Metal device, command queue, and shader library. Create one per application and pass references to kernels.

```rust
use vx_vision::Context;

let ctx = Context::new()?;
```

### Texture

`Texture` wraps a Metal texture with a known format. Textures are the primary data type for GPU operations.

```rust
// Create from pixel data
let tex = ctx.texture_gray8(&pixels, width, height)?;
let tex = ctx.texture_rgba8(&pixels, width, height)?;
let tex = ctx.texture_r32float(&data, width, height)?;

// Create empty output textures
let out = ctx.texture_output_gray8(width, height)?;

// Read results back to CPU
let pixels: Vec<u8> = tex.read_gray8();
let data: Vec<f32> = tex.read_r32float();
```

On Apple Silicon, textures live in unified memory — the CPU and GPU access the same physical bytes. There are no hidden copies.

### Kernels

Each GPU operation is a kernel struct. The pattern is always:

1. Create the kernel (compiles the Metal pipeline — do this once)
2. Call the kernel's method (dispatches GPU work)
3. Read back results

```rust
use vx_vision::kernels::gaussian::GaussianBlur;

let blur = GaussianBlur::new(&ctx)?;
let output = blur.apply(&ctx, &input, 2.0)?;  // sigma = 2.0
let result = output.read_gray8();
```

## Your First Program

Here's a complete example that loads an image, detects edges, and writes the result:

```rust
use vx_vision::Context;
use vx_vision::kernels::sobel::SobelFilter;
use vx_vision::kernels::gaussian::GaussianBlur;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize GPU context
    let ctx = Context::new()?;

    // Load image (using the `image` crate)
    let img = image::open("input.png")?.to_luma8();
    let (w, h) = img.dimensions();

    // Upload to GPU
    let texture = ctx.texture_gray8(img.as_raw(), w, h)?;

    // Blur then detect edges
    let blur = GaussianBlur::new(&ctx)?;
    let sobel = SobelFilter::new(&ctx)?;

    let blurred = blur.apply(&ctx, &texture, 1.5)?;
    let (mag, _dir) = sobel.apply(&ctx, &blurred)?;

    // Read back edge magnitude
    let edges = mag.read_r32float();

    // Convert to u8 and save
    let max_val = edges.iter().cloned().fold(0.0f32, f32::max);
    let output: Vec<u8> = edges.iter()
        .map(|&v| ((v / max_val) * 255.0) as u8)
        .collect();

    image::save_buffer("edges.png", &output, w, h, image::ColorType::L8)?;
    println!("Saved edges.png");
    Ok(())
}
```

## Texture Formats

VX supports three texture formats:

| Format | Rust type | Use case |
|---|---|---|
| `R8Unorm` (Gray8) | `Vec<u8>` | Grayscale images, masks, binary outputs |
| `RGBA8Unorm` | `Vec<u8>` (4 bytes/pixel) | Color images |
| `R32Float` | `Vec<f32>` | Edge magnitudes, scores, floating-point results |

Most kernels work on grayscale input. Use `ColorConvert` to go from RGBA to grayscale:

```rust
use vx_vision::kernels::color::ColorConvert;

let converter = ColorConvert::new(&ctx)?;
let gray = converter.rgba_to_gray(&ctx, &rgba_texture)?;
```

## Running Examples

The repository includes several runnable examples:

```bash
# Basic feature detection
cargo run --example fast_demo -- path/to/image.png

# Edge detection pipeline
cargo run --example edge_detection_demo -- path/to/image.png

# Threshold and Otsu binarization
cargo run --example threshold_demo -- path/to/image.png

# Warp and resize
cargo run --example warp_resize_demo -- path/to/image.png

# Advanced multi-kernel pipeline
cargo run --example advanced_cv_demo -- path/to/image.png

# Feature matching comparison
cargo run --example feature_matching_demo -- path/to/image.png

# Performance benchmarking with TexturePool
cargo run --example pipeline_pool_demo -- path/to/image.png
```

## Running Tests

```bash
# Run all tests (requires Metal-compatible GPU)
cargo test

# Run only kernel tests
cargo test -p vx-vision --test test_kernels

# Run a specific test
cargo test -p vx-vision --test test_kernels -- gaussian_blur
```

## Next Steps

- [API Guide](api-guide.md) — detailed reference for every kernel
- [Performance](performance.md) — TexturePool, Pipeline batching, optimization
