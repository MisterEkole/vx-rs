# Getting Started

## Requirements

- macOS on Apple Silicon (M1/M2/M3/M4) or any Mac with a Metal-capable GPU
- Rust stable toolchain
- Xcode command line tools: `xcode-select --install`

## Installation

```toml
[dependencies]
vx-vision = "0.1"
```

## Core concepts

### Context

`Context` initializes the Metal device, command queue, and shader library. Create one at startup and pass references to kernels.

```rust
let ctx = vx_vision::Context::new()?;
```

### Texture

`Texture` wraps a Metal texture with a known format. Three formats are supported:

| Format | Create from data | Create empty | Read back |
|---|---|---|---|
| R8Unorm (grayscale) | `ctx.texture_gray8(&pixels, w, h)` | `ctx.texture_output_gray8(w, h)` | `tex.read_gray8()` |
| R32Float | `ctx.texture_r32float(&data, w, h)` | `ctx.texture_output_r32float(w, h)` | `tex.read_r32float()` |
| RGBA8Unorm (color) | `ctx.texture_rgba8(&pixels, w, h)` | `ctx.texture_output_rgba8(w, h)` | `tex.read_rgba8()` |

On Apple Silicon, textures live in unified memory — no hidden copies between CPU and GPU.

### Kernels

Each GPU operation is a struct. The pattern is always:

1. **Create** the kernel (compiles the Metal pipeline — do this once)
2. **Call** the kernel method (dispatches GPU work)
3. **Read** results back

```rust
let blur = GaussianBlur::new(&ctx)?;
let output = ctx.texture_output_gray8(w, h)?;
blur.apply(&ctx, &input, &output, &GaussianConfig::default())?;
let result = output.read_gray8();
```

## First program

Load an image, blur it, detect edges:

```rust
use vx_vision::Context;
use vx_vision::kernels::gaussian::{GaussianBlur, GaussianConfig};
use vx_vision::kernels::sobel::SobelFilter;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = Context::new()?;

    let img = image::open("input.png")?.to_luma8();
    let (w, h) = img.dimensions();
    let texture = ctx.texture_gray8(img.as_raw(), w, h)?;

    let blur = GaussianBlur::new(&ctx)?;
    let sobel = SobelFilter::new(&ctx)?;

    let blurred = ctx.texture_output_gray8(w, h)?;
    blur.apply(&ctx, &texture, &blurred, &GaussianConfig::default())?;

    let result = sobel.compute(&ctx, &blurred)?;
    let edges = result.magnitude.read_r32float();

    let max_val = edges.iter().cloned().fold(0.0f32, f32::max);
    let output: Vec<u8> = edges.iter()
        .map(|&v| ((v / max_val) * 255.0) as u8)
        .collect();

    image::save_buffer("edges.png", &output, w, h, image::ColorType::L8)?;
    Ok(())
}
```

Add `image = "0.25"` to your `Cargo.toml` dependencies.

## Running examples

```bash
cargo run --example fast_demo -- path/to/image.png
cargo run --example edge_detection_demo -- path/to/image.png
cargo run --example threshold_demo -- path/to/image.png
cargo run --example advanced_cv_demo -- path/to/image.png
cargo run --example feature_matching_demo -- path/to/image.png
cargo run --example pipeline_pool_demo -- path/to/image.png
```

## Running tests

```bash
cargo test                            # everything
cargo test -p vx-vision               # kernel tests only
cargo test -p vx-vision -- gaussian   # specific test
```
