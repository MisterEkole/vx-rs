# Performance Guide

## TexturePool

GPU texture allocation is expensive. If you're processing multiple frames or running multi-stage pipelines, reuse textures instead of allocating new ones every time.

```rust
use vx_vision::{Context, TexturePool};

let ctx = Context::new()?;
let mut pool = TexturePool::new();

// First call allocates a new texture (cache miss)
let tex = pool.acquire_gray8(&ctx, 1920, 1080)?;
// ... use tex for GPU work ...

// Return to pool for reuse
pool.release(tex);

// Second call reuses the cached texture (cache hit — no allocation)
let tex = pool.acquire_gray8(&ctx, 1920, 1080)?;
```

The pool keys textures by `(width, height, format)`. A texture is only reused if all three match.

### Pool Configuration

```rust
// Default: up to 8 textures per (width, height, format) bucket
let pool = TexturePool::new();

// Custom capacity limit
let pool = TexturePool::with_capacity(4);

// Monitor pool efficiency
println!("Hit rate: {:.1}%", pool.hit_rate() * 100.0);
println!("Cached: {}", pool.cached_count());

// Free all cached textures
pool.clear();
```

### Available Formats

```rust
let gray = pool.acquire_gray8(&ctx, w, h)?;      // R8Unorm
let float = pool.acquire_r32float(&ctx, w, h)?;   // R32Float
let color = pool.acquire_rgba8(&ctx, w, h)?;       // RGBA8Unorm
```

All pool-allocated textures have `ShaderRead | ShaderWrite` usage flags so they work as both input and output for any kernel.

## Pipeline Batching

By default, each kernel dispatch creates its own command buffer and waits for completion. For multi-stage pipelines, this means N round-trips to the GPU. The `Pipeline` builder batches all dispatches into a single command buffer.

```rust
use vx_vision::Pipeline;

let mut pipe = Pipeline::begin(&ctx)?;

// Encode multiple operations into one command buffer
blur.encode(pipe.cmd_buf(), &input, &temp1)?;
sobel.encode(pipe.cmd_buf(), &temp1, &mag, &dir)?;

// Retain intermediate textures so they stay alive until GPU completes
pipe.retain(temp1);

// Submit everything at once
let retained = pipe.commit_and_wait();
// `retained` holds the intermediate textures; drop or release to pool
```

### Async Pipeline

For CPU/GPU overlap, use `commit()` + `wait()` separately:

```rust
let mut pipe = Pipeline::begin(&ctx)?;
blur.encode(pipe.cmd_buf(), &input, &output)?;
pipe.commit();

// Do CPU work while GPU executes...
process_metadata();

// Block until GPU is done
pipe.wait();
```

## Optimization Tips

### 1. Reuse kernel structs

Creating a kernel compiles the Metal pipeline. Do it once, reuse across frames:

```rust
// Do this once
let blur = GaussianBlur::new(&ctx)?;
let sobel = SobelFilter::new(&ctx)?;

// Reuse per frame
for frame in frames {
    let blurred = blur.apply(&ctx, &frame, 1.5)?;
    let (mag, _) = sobel.apply(&ctx, &blurred)?;
}
```

### 2. Avoid unnecessary readbacks

Reading results back to the CPU (`read_gray8()`, `read_r32float()`) forces a GPU synchronization. If the output feeds another GPU kernel, pass the texture directly:

```rust
// Bad: readback + re-upload between stages
let blurred = blur.apply(&ctx, &input, 1.5)?;
let pixels = blurred.read_gray8();  // unnecessary CPU roundtrip
let tex = ctx.texture_gray8(&pixels, w, h)?;
let edges = sobel.apply(&ctx, &tex)?;

// Good: pass texture directly
let blurred = blur.apply(&ctx, &input, 1.5)?;
let edges = sobel.apply(&ctx, &blurred)?;
```

### 3. Use TexturePool for repeated processing

```rust
let mut pool = TexturePool::new();

for frame in video_frames {
    let temp = pool.acquire_gray8(&ctx, w, h)?;
    // ... process ...
    pool.release(temp);
}
```

### 4. Size textures appropriately

Larger textures = more GPU threads. If you don't need full resolution, downsample first:

```rust
let pyr = PyramidBuilder::new(&ctx)?;
let levels = pyr.build(&ctx, &input, 3)?;

// Detect features on half-resolution image
let corners = fast.detect(&ctx, &levels[0], &config)?;
```

### 5. Batch with Pipeline for multi-stage processing

Single command buffer submission is faster than N individual dispatches:

```rust
// Instead of 5 separate dispatches (5 command buffers):
let a = blur.apply(&ctx, &input, 1.0)?;
let b = blur.apply(&ctx, &a, 1.0)?;
// ...

// Use Pipeline (1 command buffer):
let mut pipe = Pipeline::begin(&ctx)?;
blur.encode(pipe.cmd_buf(), &input, &temp1)?;
blur.encode(pipe.cmd_buf(), &temp1, &temp2)?;
// ...
pipe.commit_and_wait();
```

## Memory Model

On Apple Silicon (UMA), CPU and GPU share the same physical memory. VX uses `MTLStorageModeShared` for all buffers, which means:

- **No copies**: Data written by the CPU is immediately accessible to the GPU and vice versa
- **No explicit sync**: The Metal command buffer's `waitUntilCompleted()` handles synchronization
- **GpuGuard**: The `UnifiedBuffer<T>` type uses `GpuGuard<T>` to prevent CPU mutation while the buffer is in-flight on the GPU, catching race conditions at runtime

This is why VX is fast on Apple Silicon — the traditional "upload to VRAM → compute → download from VRAM" cycle is eliminated entirely.
