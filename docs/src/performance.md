# Pipeline & Performance

## Pipeline batching

By default, each kernel's sync method (`apply`, `compute`, `detect`) creates its own command buffer and waits for completion. For multi-stage pipelines, this means N GPU round-trips.

`Pipeline` batches everything into a single command buffer:

```rust
use vx_vision::Pipeline;

let pipe = Pipeline::begin(&ctx)?;
let cmd = pipe.cmd_buf();

let s1 = blur.encode(&ctx, cmd, &input, &temp1, &blur_cfg)?;
bilateral.encode(cmd, &temp1, &temp2, &bilateral_cfg)?;
morph.encode_dilate(cmd, &temp2, &output, &morph_cfg)?;

let _retained = pipe.commit_and_wait();
```

Encoded state (like `s1` above) holds intermediate textures that must outlive the command buffer.

### Which kernels support encoding?

| Encodable | Not encodable (multi-pass) |
|---|---|
| Gaussian, Bilateral, Sobel, Canny, Morphology, Threshold, Color, Warp, Integral, Dense Flow, FAST, Harris, NMS, ORB, KLT, Resize, Undistort, DoG (subtract only) | Histogram, Homography, Connected Components, Hough |

Multi-pass kernels require CPU readback between GPU passes, so they can't be batched.

## TexturePool

GPU texture allocation is expensive. Reuse textures across frames:

```rust
use vx_vision::TexturePool;

let mut pool = TexturePool::new();

for frame in frames {
    let temp = pool.acquire_gray8(&ctx, w, h)?;  // reuses cached texture
    blur.apply(&ctx, &frame, &temp, &cfg)?;
    // ... process ...
    pool.release(temp);  // return to pool
}

println!("Hit rate: {:.0}%", pool.hit_rate() * 100.0);
```

The pool keys by `(width, height, format)`. All pool textures have `ShaderRead | ShaderWrite` flags.

## Optimization tips

**Reuse kernel structs.** Creating a kernel compiles the Metal pipeline. Do it once at startup.

```rust
let blur = GaussianBlur::new(&ctx)?;  // once
for frame in frames {
    blur.apply(&ctx, &frame, &output, &cfg)?;  // reuse
}
```

**Avoid unnecessary readbacks.** `read_gray8()` forces GPU sync. If the output feeds another kernel, pass the texture directly.

**Downsample first.** Run feature detection on half-resolution images when full resolution isn't needed:

```rust
let levels = pyr.build(&ctx, &input, 3)?;
let corners = fast.detect(&ctx, &levels[0], &cfg)?;  // half-res
```

**Batch with Pipeline.** One command buffer is faster than five:

```rust
let pipe = Pipeline::begin(&ctx)?;
// encode 5 kernels into pipe.cmd_buf()
pipe.commit_and_wait();
```

## Memory model

On Apple Silicon (UMA), CPU and GPU share physical memory. VX uses `MTLStorageModeShared` — no copies, no uploads, no downloads. `waitUntilCompleted()` is the only synchronization needed.

`GpuGuard<T>` in `vx-gpu` prevents CPU reads of a `UnifiedBuffer<T>` while the GPU is using it, catching race conditions at runtime.

## Benchmarking

Run the built-in criterion benchmarks:

```bash
cargo bench -p vx-vision
```

Benchmarks include:
- FAST at 752x480 and 1920x1080
- Full FAST → Harris → NMS → ORB pipeline at both resolutions
- Pipeline vs individual dispatch comparison (3x Gaussian)
