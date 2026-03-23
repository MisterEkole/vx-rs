# Context & Texture

## Context

Entry point for all GPU operations. Holds the Metal device, command queue, and compiled shader library.

```rust
use vx_vision::Context;

let ctx = Context::new()?;
```

### Texture creation

```rust
// From pixel data (ShaderRead)
let gray  = ctx.texture_gray8(&pixels, w, h)?;
let float = ctx.texture_r32float(&data, w, h)?;
let color = ctx.texture_rgba8(&pixels, w, h)?;

// Empty output (ShaderWrite)
let out = ctx.texture_output_gray8(w, h)?;
let out = ctx.texture_output_r32float(w, h)?;
let out = ctx.texture_output_rgba8(w, h)?;

// Pipeline intermediates (ShaderRead | ShaderWrite)
let tmp = ctx.texture_intermediate_gray8(w, h)?;
let tmp = ctx.texture_intermediate_r32float(w, h)?;
```

Use `output_*` when a texture is only written to by a kernel. Use `intermediate_*` when a texture is written by one kernel and read by the next in a pipeline chain.

## Texture

Wraps a Metal texture with tracked dimensions and format.

### Readback

```rust
let pixels: Vec<u8>  = tex.read_gray8();      // R8Unorm
let data:   Vec<f32> = tex.read_r32float();    // R32Float
let pixels: Vec<u8>  = tex.read_rgba8();       // RGBA8Unorm (4 bytes/pixel)
```

Call readback only after the GPU command buffer has completed. Reading while the GPU is still writing produces undefined results.

### Properties

```rust
let w = tex.width();       // u32
let h = tex.height();      // u32
let f = tex.format();      // TextureFormat enum
```

### External textures

For AVFoundation or Core Video integration, wrap an existing Metal texture without copying:

```rust
use vx_vision::{Texture, TextureFormat};

let tex = Texture::from_metal_texture(metal_tex, w, h, TextureFormat::RGBA8Unorm);
```

## Pipeline

Batches multiple kernel dispatches into a single Metal command buffer.

```rust
use vx_vision::Pipeline;

let pipe = Pipeline::begin(&ctx)?;
let cmd = pipe.cmd_buf();

let _s1 = blur.encode(&ctx, cmd, &input, &temp, &cfg)?;
sobel.encode(&ctx, cmd, &temp)?;

let _retained = pipe.commit_and_wait();
```

Intermediate textures and encoded state must outlive the command buffer. The `commit_and_wait()` return value holds retained textures.

For CPU/GPU overlap:

```rust
let mut pipe = Pipeline::begin(&ctx)?;
blur.encode(&ctx, pipe.cmd_buf(), &input, &output, &cfg)?;
pipe.commit();       // non-blocking
// ... CPU work ...
pipe.wait();         // block until GPU done
```

## TexturePool

Recycles GPU textures by `(width, height, format)` to avoid repeated allocation.

```rust
use vx_vision::TexturePool;

let mut pool = TexturePool::new();
let tex = pool.acquire_gray8(&ctx, 1920, 1080)?;
// ... use tex ...
pool.release(tex);

// Second acquire reuses the cached texture
let tex = pool.acquire_gray8(&ctx, 1920, 1080)?;
```

All pool textures have `ShaderRead | ShaderWrite` usage flags.

```rust
let pool = TexturePool::with_capacity(4);   // max 4 per bucket
pool.hit_rate();                             // cache efficiency
pool.cached_count();                         // total cached
pool.clear();                                // free all
```

## Error handling

All fallible operations return `Result<T, vx_vision::Error>`. Error variants:

| Variant | Meaning |
|---|---|
| `DeviceNotFound` | No Metal GPU available |
| `ShaderMissing(String)` | Named shader function not in metallib |
| `PipelineCompile(String)` | Metal failed to compile a pipeline |
| `BufferAlloc { bytes }` | GPU buffer allocation failed |
| `TextureSizeMismatch` | Texture dimensions don't match |
| `InvalidConfig(String)` | Parameter out of range |
| `Gpu(String)` | Runtime GPU error |
