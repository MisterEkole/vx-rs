# Architecture

## Three-layer stack

![VX Architecture](vx_architecture_diagram.svg)

### Memory layer (`vx-gpu`)

The `vx-core/` directory, published as the `vx-gpu` crate. Manages shared GPU/CPU buffers.

- **`UnifiedBuffer<T>`** — Type-safe wrapper around `MTLBuffer` with `StorageModeShared`. Provides `write()`, `as_slice()`, `as_mut_slice()`. All element types must implement `bytemuck::Pod + Zeroable`.
- **`GpuGuard<T>`** — RAII guard that prevents CPU mutation while a buffer is in-flight on the GPU. Create before `commit()`, drop after `waitUntilCompleted()`.
- **Device helpers** — `default_device()`, `new_queue()`, `load_library_from_bytes()`.

### Kernel layer (`vx-vision`)

The `vx-vision/` directory. Contains Rust bindings for each Metal shader.

- **`Context`** — Holds the Metal device, command queue, and compiled shader library. Entry point for everything.
- **`Texture`** — GPU texture with tracked dimensions and format. Provides readback methods and zero-copy wrapping of external Metal textures.
- **`Pipeline`** — Batches multiple kernel dispatches into a single Metal command buffer.
- **`TexturePool`** — Recycles textures by `(width, height, format)` to avoid repeated allocation.
- **Kernel structs** — One per algorithm (e.g., `FastDetector`, `GaussianBlur`, `CannyDetector`). Each holds compiled `MTLComputePipelineState` objects, constructed once and reused.

## Shader-to-kernel contract

Each algorithm has two sides:

| Component | Location | Naming |
|---|---|---|
| Metal shader | `vx-vision/shaders/PascalCase.metal` | kernel function: `snake_case` |
| Rust binding | `vx-vision/src/kernels/snake_case.rs` | struct: `PascalCase` |

Example: `FastDetect.metal` defines `kernel void fast_detect(...)`, and `fast.rs` defines `FastDetector` which compiles that function into a pipeline at construction.

### Parameter structs

GPU parameter structs live in `vx-vision/src/types.rs` with `#[repr(C)]` layout. They must match the MSL struct field-by-field:

| Rust | Metal |
|---|---|
| `u32` | `uint` |
| `i32` | `int` |
| `f32` | `float` |
| `[f32; 2]` | `float2` |
| `[f32; 3]` + `_pad: f32` | `float3` (16-byte aligned) |
| `[f32; 4]` | `float4` |

Any mismatch causes silent data corruption.

## Build system

`vx-vision/build.rs` auto-discovers all `.metal` files in `vx-vision/shaders/`, compiles each to `.air` via `xcrun metal`, links into `vx.metallib` via `xcrun metallib`, and embeds it via `include_bytes!`. Adding a new `.metal` file triggers automatic recompilation.

## Thread dispatch patterns

- **2D per-pixel** (image filters): grid = `(width, height, 1)`, threadgroup computed from `threadExecutionWidth()` and `maxTotalThreadsPerThreadgroup()`
- **1D per-element** (feature operations): grid = `(n, 1, 1)`, threadgroup = `(threadExecutionWidth, 1, 1)`
- Always uses `dispatchThreads:threadsPerThreadgroup:` (non-uniform dispatch)

## Thread safety

All kernel structs, `Context`, and `Texture` implement `Send + Sync`. Metal pipeline state objects are immutable after creation. `MTLCommandQueue` is thread-safe, but each thread should create its own command buffers.

## Memory model

On Apple Silicon (UMA), CPU and GPU share physical memory. VX uses `MTLStorageModeShared` for all buffers:

- **No copies** — data written by CPU is immediately visible to GPU and vice versa
- **Synchronization** — `waitUntilCompleted()` on the command buffer is sufficient
- **Safety** — `GpuGuard<T>` prevents CPU mutation while GPU is reading
