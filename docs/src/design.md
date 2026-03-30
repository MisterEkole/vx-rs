# Design Document

Internal engineering reference. Assumes familiarity with Rust and GPU compute concepts.

---

## 1. System Overview

### Crate Topology

<svg viewBox="0 0 720 340" xmlns="http://www.w3.org/2000/svg" style="max-width:720px;font-family:monospace">
  <defs>
    <marker id="arr" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
    </marker>
  </defs>
  <!-- Application -->
  <rect x="180" y="10" width="360" height="48" rx="6" fill="#e8f5e9" stroke="#388e3c" stroke-width="1.5"/>
  <text x="360" y="30" text-anchor="middle" font-size="13" font-weight="bold" fill="#1b5e20">Application Layer</text>
  <text x="360" y="46" text-anchor="middle" font-size="10" fill="#388e3c">examples/, downstream consumers</text>
  <!-- vx-vision -->
  <rect x="130" y="90" width="460" height="58" rx="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.5"/>
  <text x="360" y="112" text-anchor="middle" font-size="13" font-weight="bold" fill="#0d47a1">vx-vision</text>
  <text x="360" y="128" text-anchor="middle" font-size="10" fill="#1565c0">Context · Texture · Pipeline · TexturePool · 28 kernels</text>
  <text x="360" y="142" text-anchor="middle" font-size="10" fill="#1565c0">42 Metal shaders · #[repr(C)] types</text>
  <!-- vx-core -->
  <rect x="180" y="185" width="360" height="48" rx="6" fill="#fff3e0" stroke="#e65100" stroke-width="1.5"/>
  <text x="360" y="205" text-anchor="middle" font-size="13" font-weight="bold" fill="#bf360c">vx-core (published as vx-gpu)</text>
  <text x="360" y="221" text-anchor="middle" font-size="10" fill="#e65100">UnifiedBuffer&lt;T&gt; · GpuGuard&lt;T&gt; · device management</text>
  <!-- External deps -->
  <rect x="30" y="275" width="130" height="36" rx="6" fill="#f3e5f5" stroke="#7b1fa2" stroke-width="1.5"/>
  <text x="95" y="298" text-anchor="middle" font-size="11" fill="#4a148c">objc2-metal</text>
  <rect x="190" y="275" width="100" height="36" rx="6" fill="#f3e5f5" stroke="#7b1fa2" stroke-width="1.5"/>
  <text x="240" y="298" text-anchor="middle" font-size="11" fill="#4a148c">objc2</text>
  <rect x="320" y="275" width="150" height="36" rx="6" fill="#f3e5f5" stroke="#7b1fa2" stroke-width="1.5"/>
  <text x="395" y="298" text-anchor="middle" font-size="11" fill="#4a148c">objc2-foundation</text>
  <rect x="500" y="275" width="110" height="36" rx="6" fill="#fce4ec" stroke="#c62828" stroke-width="1.5"/>
  <text x="555" y="298" text-anchor="middle" font-size="11" fill="#b71c1c">bytemuck</text>
  <!-- Arrows: App → vx-vision -->
  <line x1="360" y1="58" x2="360" y2="88" stroke="#333" stroke-width="1.5" marker-end="url(#arr)"/>
  <!-- vx-vision → vx-core -->
  <line x1="360" y1="148" x2="360" y2="183" stroke="#333" stroke-width="1.5" marker-end="url(#arr)"/>
  <!-- vx-core → externals -->
  <line x1="270" y1="233" x2="110" y2="273" stroke="#333" stroke-width="1.5" marker-end="url(#arr)"/>
  <line x1="320" y1="233" x2="240" y2="273" stroke="#333" stroke-width="1.5" marker-end="url(#arr)"/>
  <line x1="400" y1="233" x2="395" y2="273" stroke="#333" stroke-width="1.5" marker-end="url(#arr)"/>
  <line x1="460" y1="233" x2="540" y2="273" stroke="#333" stroke-width="1.5" marker-end="url(#arr)"/>
  <!-- vx-vision also depends on externals (dashed) -->
  <line x1="200" y1="148" x2="95" y2="273" stroke="#999" stroke-width="1" stroke-dasharray="5,3" marker-end="url(#arr)"/>
  <line x1="500" y1="148" x2="555" y2="273" stroke="#999" stroke-width="1" stroke-dasharray="5,3" marker-end="url(#arr)"/>
</svg>

Two workspace crates. `vx-vision` imports `vx-core` as `vx_gpu`. Both crates depend on `objc2-metal` for Metal bindings and `bytemuck` for safe `#[repr(C)]` transmutes. No other runtime dependencies — no `wgpu`, no `ash`, no cross-platform abstraction. Metal-only by design: this eliminates translation layers and enables direct access to Apple Silicon UMA semantics that cross-platform APIs abstract away.

---

## 2. Metal–Rust Bridge

### objc2-metal

Metal is an Objective-C framework. Its API surface is protocol-based — `MTLDevice`, `MTLCommandBuffer`, etc. are protocols, not concrete classes. The runtime returns opaque objects conforming to these protocols.

`objc2-metal` generates Rust trait definitions from Metal's protocol headers. Each protocol becomes a Rust trait. Since the concrete type behind the protocol is unknown at compile time, the binding uses type-erased wrappers:

```rust
Retained<ProtocolObject<dyn MTLComputePipelineState>>
```

Three layers here:

- **`dyn MTLComputePipelineState`** — the Metal protocol, expressed as a Rust trait object
- **`ProtocolObject<_>`** — type erasure wrapper for ObjC protocol conformance (analogous to `dyn Trait` but bridging ObjC's protocol dispatch, not Rust's vtable dispatch)
- **`Retained<_>`** — ARC-compatible smart pointer. Holds a +1 reference count. Sends `release` on drop. This is the bridge between ObjC's reference counting and Rust's ownership semantics.

`Retained` means Rust *owns* the object. When a kernel struct holds `Retained<ProtocolObject<dyn MTLComputePipelineState>>`, the pipeline state lives exactly as long as the kernel struct. No manual retain/release, no leak risk, no double-free — Rust's drop semantics handle it.

### Metal Object Roles

| Object | Created by | Role | Cost | Reuse |
|---|---|---|---|---|
| `MTLDevice` | `MTLCreateSystemDefaultDevice()` | GPU handle, factory for everything | One-time | Always |
| `MTLLibrary` | `device.newLibraryWithURL_error()` | Container of compiled shader functions | One-time (loads metallib) | Always |
| `MTLComputePipelineState` | `device.newComputePipelineStateWithFunction_error()` | Compiled, optimized kernel ready for dispatch | Expensive (shader compilation, register allocation, occupancy calculation) | Always |
| `MTLCommandQueue` | `device.newCommandQueue()` | Serial scheduler that submits command buffers to GPU | One-time | Always |
| `MTLCommandBuffer` | `queue.commandBuffer()` | Single batch of GPU work | Cheap (pool-allocated internally by Metal) | Never (one-shot) |
| `MTLComputeCommandEncoder` | `cmd_buf.computeCommandEncoder()` | Records bind + dispatch commands into a command buffer | Cheap | Never (one-shot) |

The key insight: objects above the line (Device through Queue) are created once and stored for app lifetime. Objects below the line (CommandBuffer, Encoder) are created per-dispatch and discarded after use. This is why kernel structs store pipelines but create command buffers on every call.

### Send + Sync

ObjC objects don't carry `Send`/`Sync` in Rust's type system. Every kernel struct, `Context`, and `Texture` needs explicit unsafe impls:

```rust
unsafe impl Send for FastDetector {}
unsafe impl Sync for FastDetector {}
```

**Justification:** Metal pipeline states are immutable after creation — Apple documents them as thread-safe. `MTLDevice` and `MTLCommandQueue` are thread-safe. `MTLCommandBuffer` is *not* thread-safe, but it's never stored in a struct — it's created and consumed within a single method call.

Without these impls, kernel structs couldn't be stored in `Arc`, passed across threads, or used in async contexts.

### GPU Command Lifecycle

<svg viewBox="0 0 780 130" xmlns="http://www.w3.org/2000/svg" style="max-width:780px;font-family:monospace">
  <defs>
    <marker id="a2" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#333"/>
    </marker>
  </defs>
  <rect x="5" y="30" width="80" height="36" rx="5" fill="#fff3e0" stroke="#e65100" stroke-width="1.5"/>
  <text x="45" y="52" text-anchor="middle" font-size="10" fill="#bf360c">Device</text>
  <rect x="115" y="30" width="80" height="36" rx="5" fill="#fff3e0" stroke="#e65100" stroke-width="1.5"/>
  <text x="155" y="52" text-anchor="middle" font-size="10" fill="#bf360c">Queue</text>
  <rect x="225" y="30" width="90" height="36" rx="5" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.5"/>
  <text x="270" y="52" text-anchor="middle" font-size="10" fill="#0d47a1">CmdBuffer</text>
  <rect x="345" y="30" width="80" height="36" rx="5" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.5"/>
  <text x="385" y="52" text-anchor="middle" font-size="10" fill="#0d47a1">Encoder</text>
  <rect x="455" y="30" width="90" height="36" rx="5" fill="#e8f5e9" stroke="#388e3c" stroke-width="1.5"/>
  <text x="500" y="46" text-anchor="middle" font-size="9" fill="#1b5e20">setPipeline</text>
  <text x="500" y="58" text-anchor="middle" font-size="9" fill="#1b5e20">setBuffers</text>
  <rect x="575" y="30" width="80" height="36" rx="5" fill="#e8f5e9" stroke="#388e3c" stroke-width="1.5"/>
  <text x="615" y="52" text-anchor="middle" font-size="10" fill="#1b5e20">Dispatch</text>
  <rect x="685" y="30" width="80" height="36" rx="5" fill="#fce4ec" stroke="#c62828" stroke-width="1.5"/>
  <text x="725" y="46" text-anchor="middle" font-size="9" fill="#b71c1c">commit()</text>
  <text x="725" y="58" text-anchor="middle" font-size="9" fill="#b71c1c">wait()</text>
  <!-- Arrows -->
  <line x1="85" y1="48" x2="113" y2="48" stroke="#333" stroke-width="1.2" marker-end="url(#a2)"/>
  <line x1="195" y1="48" x2="223" y2="48" stroke="#333" stroke-width="1.2" marker-end="url(#a2)"/>
  <line x1="315" y1="48" x2="343" y2="48" stroke="#333" stroke-width="1.2" marker-end="url(#a2)"/>
  <line x1="425" y1="48" x2="453" y2="48" stroke="#333" stroke-width="1.2" marker-end="url(#a2)"/>
  <line x1="545" y1="48" x2="573" y2="48" stroke="#333" stroke-width="1.2" marker-end="url(#a2)"/>
  <line x1="655" y1="48" x2="683" y2="48" stroke="#333" stroke-width="1.2" marker-end="url(#a2)"/>
  <!-- Lifetime labels -->
  <text x="45" y="84" text-anchor="middle" font-size="8" fill="#666">app lifetime</text>
  <text x="155" y="84" text-anchor="middle" font-size="8" fill="#666">app lifetime</text>
  <text x="270" y="84" text-anchor="middle" font-size="8" fill="#666">per batch</text>
  <text x="385" y="84" text-anchor="middle" font-size="8" fill="#666">per kernel</text>
  <text x="500" y="84" text-anchor="middle" font-size="8" fill="#666">bind args</text>
  <text x="615" y="84" text-anchor="middle" font-size="8" fill="#666">launch grid</text>
  <text x="725" y="84" text-anchor="middle" font-size="8" fill="#666">fence + block</text>
  <!-- Scope bracket -->
  <line x1="225" y1="100" x2="765" y2="100" stroke="#1565c0" stroke-width="1" stroke-dasharray="4,2"/>
  <text x="495" y="115" text-anchor="middle" font-size="8" fill="#1565c0">per-dispatch scope (created and consumed in one method call)</text>
</svg>

The Rust code mapping each step:

```rust
// ── App lifetime (Context::new) ──────────────────────────────────
let device  = MTLCreateSystemDefaultDevice();           // GPU handle
let queue   = device.newCommandQueue();                 // submission queue
let library = device.newLibraryWithURL_error(&url);     // compiled shaders

// ── App lifetime (Kernel::new) ───────────────────────────────────
let func     = library.newFunctionWithName(ns_string!("fast_detect"));  // function handle
let pipeline = device.newComputePipelineStateWithFunction_error(&func); // compiled kernel
// ↑ This is the expensive step: Metal compiles the function to GPU ISA,
//   determines register usage, calculates max occupancy. ~1-10ms per kernel.

// ── Per dispatch (Kernel::detect / apply / compute) ──────────────
let cmd_buf = queue.commandBuffer();                    // lightweight, pool-allocated
let encoder = cmd_buf.computeCommandEncoder();          // begins a compute pass

encoder.setComputePipelineState(&pipeline);             // which kernel to run
encoder.setTexture_atIndex(Some(tex.raw()), 0);         // [[texture(0)]] in shader
encoder.setBuffer_offset_atIndex(                       // [[buffer(0)]] in shader
    Some(buf.metal_buffer()), 0, 0);
encoder.setBytes_length_atIndex(                        // [[buffer(2)]] — inline constant data
    NonNull::new_unchecked(params_ptr), size, 2);       // memcpy'd into argument buffer
encoder.dispatchThreads_threadsPerThreadgroup(grid, tg); // launch the grid
encoder.endEncoding();                                   // seal this compute pass

cmd_buf.commit();              // submit to GPU — non-blocking, returns immediately
cmd_buf.waitUntilCompleted();  // block CPU until GPU finishes this command buffer
// After this point: all buffer/texture writes by the GPU are visible to CPU.
```

`setBytes` copies the struct directly into Metal's argument buffer (max 4KB), avoiding a `MTLBuffer` allocation for small constant data. Every kernel uses this for its `Params` struct. For large data (keypoint arrays, descriptors), `setBuffer` is required — it binds an existing `MTLBuffer` by reference.

---

## 3. Memory Model

### Apple Silicon UMA

Apple Silicon uses Unified Memory Architecture — CPU and GPU share the same physical DRAM. No PCIe bus. No DMA copies. A pointer to shared memory is valid on both processors.

`MTLStorageModeShared` exposes this: one allocation, one virtual address space, zero-copy access from both CPU and GPU. This is the only storage mode vx-rs uses.

**Trade-off:** On discrete GPUs (NVIDIA/AMD), `StorageModeShared` would require PCIe transfers on every access. On Apple Silicon, it's free — both CPU and GPU access the same cache hierarchy. This is a deliberate platform bet: zero-copy UMA is the core performance advantage that makes the "no cross-platform" design worthwhile.

**Synchronization model:** After `cmd_buf.waitUntilCompleted()`, all GPU writes are visible to CPU. No explicit cache flush needed — Metal's command buffer completion acts as a full memory fence. Before that fence, CPU reads may see stale data (writes can be in GPU caches or reorder buffers).

### UnifiedBuffer\<T\>

`vx-core/src/buffer.rs`

```rust
pub struct UnifiedBuffer<T: Pod> {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,  // the Metal buffer (+1 refcount)
    count: usize,                                   // number of T elements
    in_flight: Arc<AtomicBool>,                     // prevents CPU mutation during GPU work
    _marker: PhantomData<T>,                        // carries T without owning one
}
```

The `Pod` bound (`bytemuck::Pod`) guarantees the type has no padding traps, no invalid bit patterns, and is safe to interpret from arbitrary bytes. Metal writes raw bytes into the buffer — `Pod` ensures the CPU can safely interpret them as `T` after GPU completion. Every `#[repr(C)]` struct in `types.rs` derives `Pod`.

**Allocation:** `newBufferWithLength_options(count * size_of::<T>(), StorageModeShared)`. Metal zero-initializes shared buffers on allocation. This is relied upon — atomic counter buffers start at zero by default (though kernels explicitly write `[0u32]` before dispatch for clarity).

**CPU access paths:**

- `as_slice() -> &[T]` — casts `MTLBuffer.contents()` pointer to `&[T]`. No copy. The slice directly aliases GPU-visible memory.
- `as_mut_slice() -> &mut [T]` — same, but asserts `!in_flight`. Panics if a `GpuGuard` exists. This prevents writing to memory the GPU is reading from.
- `write(&[T])` — `copy_from_slice` into the buffer. Panics if in-flight.
- `to_vec() -> Vec<T>` — copies out. Use after GPU completion for safe owned data.

### GpuGuard\<T\>

`vx-core/src/buffer.rs`

RAII guard that blocks CPU mutation while the buffer is in-flight on the GPU.

```rust
pub struct GpuGuard<T: Pod> {
    in_flight: Arc<AtomicBool>,  // shared with the UnifiedBuffer
    _marker: PhantomData<T>,
}

// Created: buf.gpu_guard() → sets in_flight = true (Release)
// Dropped: sets in_flight = false (Release)
// Checked: as_mut_slice() loads in_flight (Acquire) → panics if true
```

**Atomic ordering:**

- `store(true, Release)` on guard creation — ensures all CPU writes to the buffer *before* creating the guard are visible before the flag becomes true. The GPU sees consistent data.
- `load(Acquire)` in `as_mut_slice()` — ensures the CPU doesn't reorder reads *after* the check to before it. If the flag is false, the GPU is done and all its writes are visible.
- `store(false, Release)` on drop — pairs with the Acquire in `as_mut_slice()` to establish happens-before.

This is not a mutex — it's a one-way gate. No contention, no blocking, just a panic if the contract is violated. The real synchronization is `waitUntilCompleted()` on the command buffer.

### Memory Lifecycle

<svg viewBox="0 0 720 110" xmlns="http://www.w3.org/2000/svg" style="max-width:720px;font-family:monospace">
  <defs>
    <marker id="a3" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#333"/>
    </marker>
  </defs>
  <rect x="5" y="25" width="70" height="30" rx="4" fill="#fff3e0" stroke="#e65100"/>
  <text x="40" y="44" text-anchor="middle" font-size="9" fill="#bf360c">Alloc</text>
  <rect x="100" y="25" width="70" height="30" rx="4" fill="#fff3e0" stroke="#e65100"/>
  <text x="135" y="44" text-anchor="middle" font-size="9" fill="#bf360c">Write</text>
  <rect x="195" y="25" width="75" height="30" rx="4" fill="#fce4ec" stroke="#c62828"/>
  <text x="232" y="44" text-anchor="middle" font-size="9" fill="#b71c1c">Guard</text>
  <rect x="295" y="25" width="80" height="30" rx="4" fill="#e3f2fd" stroke="#1565c0"/>
  <text x="335" y="44" text-anchor="middle" font-size="9" fill="#0d47a1">Dispatch</text>
  <rect x="400" y="25" width="70" height="30" rx="4" fill="#e3f2fd" stroke="#1565c0"/>
  <text x="435" y="44" text-anchor="middle" font-size="9" fill="#0d47a1">Wait</text>
  <rect x="495" y="25" width="90" height="30" rx="4" fill="#fce4ec" stroke="#c62828"/>
  <text x="540" y="44" text-anchor="middle" font-size="9" fill="#b71c1c">Drop Guard</text>
  <rect x="610" y="25" width="80" height="30" rx="4" fill="#e8f5e9" stroke="#388e3c"/>
  <text x="650" y="44" text-anchor="middle" font-size="9" fill="#1b5e20">Read</text>
  <!-- Arrows -->
  <line x1="75" y1="40" x2="98" y2="40" stroke="#333" marker-end="url(#a3)"/>
  <line x1="170" y1="40" x2="193" y2="40" stroke="#333" marker-end="url(#a3)"/>
  <line x1="270" y1="40" x2="293" y2="40" stroke="#333" marker-end="url(#a3)"/>
  <line x1="375" y1="40" x2="398" y2="40" stroke="#333" marker-end="url(#a3)"/>
  <line x1="470" y1="40" x2="493" y2="40" stroke="#333" marker-end="url(#a3)"/>
  <line x1="585" y1="40" x2="608" y2="40" stroke="#333" marker-end="url(#a3)"/>
  <!-- Zone labels -->
  <text x="40" y="72" text-anchor="middle" font-size="8" fill="#999">CPU</text>
  <text x="135" y="72" text-anchor="middle" font-size="8" fill="#999">CPU</text>
  <text x="232" y="72" text-anchor="middle" font-size="8" fill="#b71c1c">lock</text>
  <text x="335" y="72" text-anchor="middle" font-size="8" fill="#1565c0">GPU</text>
  <text x="435" y="72" text-anchor="middle" font-size="8" fill="#1565c0">fence</text>
  <text x="540" y="72" text-anchor="middle" font-size="8" fill="#b71c1c">unlock</text>
  <text x="650" y="72" text-anchor="middle" font-size="8" fill="#999">CPU</text>
  <!-- Danger zone -->
  <rect x="195" y="82" width="375" height="16" rx="3" fill="#fff8e1" stroke="#f9a825" stroke-width="0.5"/>
  <text x="382" y="93" text-anchor="middle" font-size="8" fill="#f57f17">CPU mutation panics in this window</text>
</svg>

**Invariant:** Guard created *before* commit. Dropped *after* `waitUntilCompleted`. Reversing this allows CPU reads of partially-written GPU data — silent corruption, not a crash.

---

## 4. Texture Subsystem

`vx-vision/src/texture.rs`

```rust
pub struct Texture {
    raw: Retained<ProtocolObject<dyn MTLTextureTrait>>,
    width: u32,
    height: u32,
    format: TextureFormat,
}
```

Textures are used over buffers for image data because Metal's texture hardware provides spatial locality optimization (tiled/swizzled memory layout), hardware bilinear interpolation, and automatic `[0,255] → [0.0,1.0]` normalization on R8Unorm reads. A buffer has linear memory layout — worse cache behavior for 2D access patterns like convolution kernels.

### Formats

| Format | `MTLPixelFormat` | Bytes/px | Readback | Shader behavior |
|---|---|---|---|---|
| `R8Unorm` | `R8Unorm` | 1 | `read_gray8() → Vec<u8>` | `image.read(gid).r` returns `[0.0, 1.0]` |
| `R32Float` | `R32Float` | 4 | `read_r32float() → Vec<f32>` | `image.read(gid).r` returns raw float |
| `RGBA8Unorm` | `RGBA8Unorm` | 4 | `read_rgba8() → Vec<u8>` | `.rgba` returns 4 normalized channels |

`R8Unorm` normalization matters: FAST multiplies by 255 to get integer intensity for threshold comparison. Sobel operates directly in `[0,1]` space. Every shader must account for the format it reads.

### Usage Flags

| Role | Flag | Why |
|---|---|---|
| Input | `ShaderRead` | GPU reads, CPU uploads via `replaceRegion`. Cheapest — Metal can optimize read-only layout. |
| Output | `ShaderWrite` | GPU writes. Cannot be sampled in the same pass. |
| Intermediate | `ShaderRead \| ShaderWrite` | Read in one pass, written in another. Required for multi-pass kernels (Gaussian H→V, Sobel→magnitude). |

Writing to a `ShaderRead`-only texture is undefined behavior — no GPU error, just corrupted output. Metal does not validate usage flags at dispatch time.

### Upload & Readback

Both use `MTLTexture`'s `replaceRegion` (upload) and `getBytes` (readback). These are synchronous CPU-side memcpy operations. `bytesPerRow` must match: `width * 1` for R8, `width * 4` for R32Float and RGBA8.

**Performance note:** These operations copy data. For input textures, the copy happens once at creation. For output readback, it happens after GPU completion. In a real-time pipeline, prefer keeping data in textures across kernel stages rather than reading back to CPU between stages — use `Pipeline` for this.

### TexturePool

`vx-vision/src/pool.rs`

```rust
pub struct TexturePool {
    buckets: HashMap<(u32, u32, TextureFormat), Vec<Texture>>,
    max_per_bucket: usize,  // default: 8
}
```

`MTLTexture` allocation is non-trivial — Metal must find contiguous GPU memory, configure tiling parameters, and set up page tables. In a real-time pipeline processing 30fps video, allocating and freeing intermediate textures per frame wastes time.

The pool caches by `(width, height, format)`. `acquire()` pops from the matching bucket or allocates fresh. `release()` pushes back. Pool textures always have `ShaderRead | ShaderWrite` — they may serve as either input or output in different pipeline stages.

`hit_rate()` tracks effectiveness. If it's low, the pipeline is using too many distinct texture sizes.

---

## 5. Context & Pipeline

### Context

`vx-vision/src/context.rs`

```rust
pub struct Context {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
}
```

Single entry point. `Context::new()` does three things:

1. `MTLCreateSystemDefaultDevice()` — gets the system GPU
2. `device.newCommandQueue()` — creates the submission queue
3. Loads the embedded metallib (see [Build System](#9-build-system))

`device()`, `queue()`, `library()` are `pub(crate)` — kernel code uses them internally, but downstream consumers only interact through `Context` and kernel structs.

One queue per `Context`. Metal command queues are serial — command buffers submitted to the same queue execute in order. This simplifies synchronization: no explicit fences between dependent dispatches. For parallel kernel execution, create multiple `Context` instances (each with its own queue).

### Pipeline

`vx-vision/src/pipeline.rs`

```rust
pub struct Pipeline {
    cmd_buf: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    retained: Vec<Texture>,
    committed: bool,
}
```

Batches multiple kernel dispatches into a single `MTLCommandBuffer`. This matters because command buffer creation has overhead — Metal must acquire a buffer from its internal pool, set up completion handlers, and track resources.

**Texture retention problem:** When a kernel's `encode()` creates an intermediate texture, that texture would be dropped when `encode()` returns. But the GPU hasn't run yet — the command buffer only records commands, it doesn't execute them until `commit()`. If the texture is freed, the GPU reads garbage.

`retain(tex)` moves the texture into the pipeline's `Vec<Texture>`, keeping it alive until `commit_and_wait()` returns.

### Pipeline Batching Flow

<svg viewBox="0 0 700 150" xmlns="http://www.w3.org/2000/svg" style="max-width:700px;font-family:monospace">
  <defs>
    <marker id="a4" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#333"/>
    </marker>
  </defs>
  <rect x="10" y="10" width="680" height="130" rx="8" fill="none" stroke="#1565c0" stroke-width="1.5" stroke-dasharray="6,3"/>
  <text x="20" y="28" font-size="11" fill="#1565c0" font-weight="bold">Pipeline (single MTLCommandBuffer)</text>
  <rect x="25" y="45" width="70" height="35" rx="4" fill="#e3f2fd" stroke="#1565c0"/>
  <text x="60" y="66" text-anchor="middle" font-size="9" fill="#0d47a1">begin()</text>
  <rect x="120" y="45" width="130" height="35" rx="4" fill="#e8f5e9" stroke="#388e3c"/>
  <text x="185" y="60" text-anchor="middle" font-size="9" fill="#1b5e20">gaussian.encode()</text>
  <text x="185" y="72" text-anchor="middle" font-size="8" fill="#388e3c">encoder 1 → endEncoding</text>
  <rect x="275" y="45" width="80" height="35" rx="4" fill="#fff3e0" stroke="#e65100"/>
  <text x="315" y="66" text-anchor="middle" font-size="9" fill="#bf360c">retain(tex)</text>
  <rect x="380" y="45" width="120" height="35" rx="4" fill="#e8f5e9" stroke="#388e3c"/>
  <text x="440" y="60" text-anchor="middle" font-size="9" fill="#1b5e20">fast.encode()</text>
  <text x="440" y="72" text-anchor="middle" font-size="8" fill="#388e3c">encoder 2 → endEncoding</text>
  <rect x="525" y="45" width="140" height="35" rx="4" fill="#fce4ec" stroke="#c62828"/>
  <text x="595" y="66" text-anchor="middle" font-size="9" fill="#b71c1c">commit_and_wait()</text>
  <line x1="95" y1="62" x2="118" y2="62" stroke="#333" marker-end="url(#a4)"/>
  <line x1="250" y1="62" x2="273" y2="62" stroke="#333" marker-end="url(#a4)"/>
  <line x1="355" y1="62" x2="378" y2="62" stroke="#333" marker-end="url(#a4)"/>
  <line x1="500" y1="62" x2="523" y2="62" stroke="#333" marker-end="url(#a4)"/>
  <text x="350" y="110" text-anchor="middle" font-size="9" fill="#666">Multiple encoders within one command buffer. Metal executes passes sequentially.</text>
  <text x="350" y="123" text-anchor="middle" font-size="9" fill="#666">Single commit/wait amortizes submission overhead across all passes.</text>
</svg>

---

## 6. Kernel Taxonomy

Every kernel struct follows the same construction pattern:

1. `new(&Context)` — compile pipeline(s) from the metallib. Store as `Retained<...>`.
2. Sync method (`detect` / `apply` / `compute`) — allocate per-dispatch resources, encode, commit, wait, readback.
3. `encode(...)` — record commands without committing. For `Pipeline` batching.
4. `unsafe impl Send + Sync` — pipeline states are immutable, thread-safe.

The difference between kernel categories is what goes in and what comes out.

### 6.1 Buffer-Output Kernels

Variable-length structured output. The shader decides at runtime how many results to emit.

**Mechanism:** A `device atomic_uint*` counter buffer starts at zero. Each thread that produces a result atomically increments the counter and writes to that slot:

```metal
uint slot = atomic_fetch_add_explicit(count, 1, memory_order_relaxed);
if (slot < params.max_corners) {
    corners[slot] = result;
}
```

`memory_order_relaxed` is sufficient — we only need the atomic increment to be unique, not ordered relative to other atomics. The slot assignment is the synchronization.

An alternative is prefix scan (parallel exclusive scan) for deterministic output ordering, but that requires two passes and shared memory coordination. Atomic append is one pass, simpler, and sufficient — output ordering isn't needed since downstream kernels (Harris, NMS) operate on the full set regardless of order.

<svg viewBox="0 0 700 140" xmlns="http://www.w3.org/2000/svg" style="max-width:700px;font-family:monospace">
  <defs>
    <marker id="a5" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#333"/>
    </marker>
  </defs>
  <rect x="10" y="35" width="100" height="40" rx="5" fill="#e3f2fd" stroke="#1565c0"/>
  <text x="60" y="52" text-anchor="middle" font-size="10" fill="#0d47a1">Texture</text>
  <text x="60" y="65" text-anchor="middle" font-size="8" fill="#1565c0">R8Unorm</text>
  <rect x="150" y="20" width="170" height="70" rx="5" fill="#e8f5e9" stroke="#388e3c"/>
  <text x="235" y="42" text-anchor="middle" font-size="10" font-weight="bold" fill="#1b5e20">Compute Shader</text>
  <text x="235" y="56" text-anchor="middle" font-size="8" fill="#388e3c">2D grid: 1 thread/pixel</text>
  <text x="235" y="68" text-anchor="middle" font-size="8" fill="#388e3c">or 1D: 1 thread/element</text>
  <text x="235" y="80" text-anchor="middle" font-size="8" fill="#388e3c">atomic_fetch_add → slot</text>
  <rect x="365" y="20" width="120" height="30" rx="5" fill="#fff3e0" stroke="#e65100"/>
  <text x="425" y="40" text-anchor="middle" font-size="9" fill="#bf360c">Buffer&lt;T&gt; [N]</text>
  <rect x="365" y="60" width="120" height="30" rx="5" fill="#fce4ec" stroke="#c62828"/>
  <text x="425" y="80" text-anchor="middle" font-size="9" fill="#b71c1c">atomic_uint count</text>
  <rect x="530" y="35" width="130" height="40" rx="5" fill="#f3e5f5" stroke="#7b1fa2"/>
  <text x="595" y="52" text-anchor="middle" font-size="9" fill="#4a148c">CPU Readback</text>
  <text x="595" y="65" text-anchor="middle" font-size="8" fill="#7b1fa2">buf[..count].to_vec()</text>
  <line x1="110" y1="55" x2="148" y2="55" stroke="#333" marker-end="url(#a5)"/>
  <line x1="320" y1="40" x2="363" y2="35" stroke="#333" marker-end="url(#a5)"/>
  <line x1="320" y1="65" x2="363" y2="75" stroke="#333" marker-end="url(#a5)"/>
  <line x1="485" y1="55" x2="528" y2="55" stroke="#333" marker-end="url(#a5)"/>
  <rect x="150" y="105" width="100" height="24" rx="4" fill="#fafafa" stroke="#bbb"/>
  <text x="200" y="121" text-anchor="middle" font-size="8" fill="#666">constant Params&amp;</text>
  <line x1="200" y1="105" x2="200" y2="92" stroke="#999" stroke-dasharray="3,2" marker-end="url(#a5)"/>
</svg>

**Annotated walkthrough: `FastDetector::detect()`** (`vx-vision/src/kernels/fast.rs`)

```rust
pub fn detect(&self, ctx: &Context, texture: &Texture, config: &FastDetectConfig)
    -> Result<FastDetectResult>
{
    let w = texture.width();
    let h = texture.height();

    // Pre-allocate max capacity. The GPU fills [0..actual_count].
    // Over-allocation is cheap on UMA — no copy, just page table entries.
    let corner_buf = UnifiedBuffer::<CornerPoint>::new(ctx.device(), config.max_corners as usize)?;
    let mut count_buf = UnifiedBuffer::<u32>::new(ctx.device(), 1)?;
    count_buf.write(&[0u32]);  // Zero the atomic counter. Forgetting this → stale count from
                                // previous dispatch or random memory, causing buffer overread.

    // Build the params struct. Field order must exactly match FASTParams in FastDetect.metal.
    let params = FASTParams { threshold: config.threshold, max_corners: config.max_corners,
                              width: w, height: h };

    // Guards BEFORE dispatch — marks buffers as in-flight.
    let _corner_guard = corner_buf.gpu_guard();
    let _count_guard  = count_buf.gpu_guard();

    // Create command buffer and encoder.
    let cmd_buf = ctx.queue().commandBuffer()
        .ok_or(Error::Gpu("failed to create command buffer".into()))?;
    let encoder = cmd_buf.computeCommandEncoder()
        .ok_or(Error::Gpu("failed to create compute encoder".into()))?;

    // Bind pipeline, textures, buffers, params.
    // Index arguments (0, 1, 2) must match [[texture(0)]], [[buffer(0)]], [[buffer(1)]], [[buffer(2)]]
    // in the .metal file. A mismatch binds the wrong data — silent corruption, not a crash.
    encoder.setComputePipelineState(&self.pipeline);
    encoder.setTexture_atIndex(Some(texture.raw()), 0);
    encoder.setBuffer_offset_atIndex(Some(corner_buf.metal_buffer()), 0, 0);
    encoder.setBuffer_offset_atIndex(Some(count_buf.metal_buffer()), 0, 1);
    encoder.setBytes_length_atIndex(
        NonNull::new_unchecked(&params as *const _ as *mut c_void),
        mem::size_of::<FASTParams>(), 2);

    // 2D dispatch: one thread per pixel. threadExecutionWidth is the SIMD width (32 on Apple Silicon).
    // maxTotalThreadsPerThreadgroup is typically 1024. So threadgroup = 32x32 = 1024 threads.
    // Edge threadgroups are partial — the shader's bounds check handles out-of-range gids.
    let tew = self.pipeline.threadExecutionWidth();
    let max_tg = self.pipeline.maxTotalThreadsPerThreadgroup();
    let grid = MTLSize { width: w as usize, height: h as usize, depth: 1 };
    let tg = MTLSize { width: tew, height: (max_tg / tew).max(1), depth: 1 };
    encoder.dispatchThreads_threadsPerThreadgroup(grid, tg);
    encoder.endEncoding();

    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();  // blocks until GPU finishes

    // Drop guards → unlocks buffers for CPU access.
    drop(_corner_guard);
    drop(_count_guard);

    // Read actual count, clamp to max (shader may have over-incremented the atomic
    // if more corners found than buffer capacity), slice the buffer.
    let n = (count_buf.as_slice()[0] as usize).min(config.max_corners as usize);
    Ok(FastDetectResult { corners: corner_buf.as_slice()[..n].to_vec() })
}
```

**Buffer-output kernel catalog:**

| Kernel | Struct | Shader | Dispatch | Output type |
|---|---|---|---|---|
| FAST-9 | `FastDetector` | `fast_detect` | 2D per-pixel | `Vec<CornerPoint>` |
| Harris | `HarrisScorer` | `harris_response` | 1D per-corner | `Vec<CornerPoint>` (scored) |
| NMS | `NmsSuppressor` | `nms_suppress` | 1D per-corner | `Vec<CornerPoint>` (filtered) |
| ORB | `OrbDescriptor` | `orb_compute` | 1D per-keypoint | `Vec<ORBOutput>` |
| Matcher | `BruteMatcher` | `hamming_distance` + `extract_matches` | 2D + 1D | `Vec<MatchResult>` |
| StereoMatch | `StereoMatcher` | `stereo_match` | 2D | `Vec<StereoMatchResult>` |
| Histogram | `HistogramComputer` | `histogram_compute` | 2D per-pixel | `Vec<u32>` (256 bins) |
| Hough | `HoughDetector` | `hough_vote` + `hough_peaks` | 2D + 1D | `Vec<HoughLine>` |
| Homography | `HomographyScorer` | `score_homography` | 1D per-point | `Vec<ScoreResult>` |

### 6.2 Texture-to-Texture Kernels

Fixed-size output. One output pixel per input pixel (or per output pixel for resize/warp). No atomic counters, no variable-length output.

<svg viewBox="0 0 620 100" xmlns="http://www.w3.org/2000/svg" style="max-width:620px;font-family:monospace">
  <defs>
    <marker id="a6" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#333"/>
    </marker>
  </defs>
  <rect x="10" y="25" width="110" height="40" rx="5" fill="#e3f2fd" stroke="#1565c0"/>
  <text x="65" y="42" text-anchor="middle" font-size="10" fill="#0d47a1">Input Texture</text>
  <text x="65" y="55" text-anchor="middle" font-size="8" fill="#1565c0">[[texture(0)]]</text>
  <rect x="170" y="15" width="170" height="60" rx="5" fill="#e8f5e9" stroke="#388e3c"/>
  <text x="255" y="38" text-anchor="middle" font-size="10" font-weight="bold" fill="#1b5e20">Compute Shader</text>
  <text x="255" y="52" text-anchor="middle" font-size="8" fill="#388e3c">2D grid, 1 thread/pixel</text>
  <text x="255" y="64" text-anchor="middle" font-size="8" fill="#388e3c">read → transform → write</text>
  <rect x="390" y="25" width="120" height="40" rx="5" fill="#fff3e0" stroke="#e65100"/>
  <text x="450" y="42" text-anchor="middle" font-size="10" fill="#bf360c">Output Texture</text>
  <text x="450" y="55" text-anchor="middle" font-size="8" fill="#e65100">[[texture(1)]]</text>
  <line x1="120" y1="45" x2="168" y2="45" stroke="#333" marker-end="url(#a6)"/>
  <line x1="340" y1="45" x2="388" y2="45" stroke="#333" marker-end="url(#a6)"/>
</svg>

**Annotated walkthrough: `GaussianBlur::apply()`** (`vx-vision/src/kernels/gaussian.rs`)

Separable 2-pass convolution. A 2D Gaussian with radius `r` requires `(2r+1)^2` reads per pixel. Separating into H+V passes reduces to `2*(2r+1)` reads — O(r) instead of O(r²).

```rust
pub fn apply(&self, ctx: &Context, input: &Texture, output: &Texture,
             config: &GaussianConfig) -> Result<()>
{
    let w = input.width();
    let h = input.height();

    // Intermediate texture for horizontal pass output / vertical pass input.
    // Must be ShaderRead|ShaderWrite — written by pass 1, read by pass 2.
    // R32Float avoids precision loss during intermediate accumulation.
    let intermediate = Texture::intermediate_r32float(ctx.device(), w, h)?;

    let params = GaussianParams { width: w, height: h, sigma: config.sigma, radius: config.radius };
    let cmd_buf = ctx.queue().commandBuffer()?;

    // Pass 1: horizontal blur. Each thread reads (2*radius+1) horizontal neighbors.
    // Clamp-to-edge boundary: shader does clamp(x + dx, 0, width-1).
    // Weights computed inline: exp(-dx²/(2σ²)). Not precomputed — ALU is faster than
    // the memory load that a weight LUT would require at these kernel sizes.
    {
        let enc = cmd_buf.computeCommandEncoder()?;
        enc.setComputePipelineState(&self.h_pipeline);
        enc.setTexture_atIndex(Some(input.raw()), 0);        // source
        enc.setTexture_atIndex(Some(intermediate.raw()), 1);  // dest
        enc.setBytes_length_atIndex(/* params */, /* size */, 0);
        enc.dispatchThreads_threadsPerThreadgroup(grid_2d(w, h), tg_2d(&self.h_pipeline));
        enc.endEncoding();  // seal pass 1
    }

    // Pass 2: vertical blur on the horizontally-blurred intermediate.
    // Metal guarantees pass 2 sees pass 1's writes — encoders within a command buffer
    // execute in order with implicit barriers between them.
    {
        let enc = cmd_buf.computeCommandEncoder()?;
        enc.setComputePipelineState(&self.v_pipeline);
        enc.setTexture_atIndex(Some(intermediate.raw()), 0);  // source (pass 1 output)
        enc.setTexture_atIndex(Some(output.raw()), 1);         // final dest
        enc.setBytes_length_atIndex(/* params */, /* size */, 0);
        enc.dispatchThreads_threadsPerThreadgroup(grid_2d(w, h), tg_2d(&self.v_pipeline));
        enc.endEncoding();
    }

    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();
    // intermediate texture dropped here — its only purpose was the H→V handoff.
    Ok(())
}
```

**Multi-pass flow:**

<svg viewBox="0 0 700 100" xmlns="http://www.w3.org/2000/svg" style="max-width:700px;font-family:monospace">
  <defs>
    <marker id="a7" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#333"/>
    </marker>
  </defs>
  <rect x="10" y="30" width="80" height="35" rx="4" fill="#e3f2fd" stroke="#1565c0"/>
  <text x="50" y="51" text-anchor="middle" font-size="9" fill="#0d47a1">Input</text>
  <rect x="130" y="30" width="120" height="35" rx="4" fill="#e8f5e9" stroke="#388e3c"/>
  <text x="190" y="45" text-anchor="middle" font-size="9" fill="#1b5e20">Pass 1 (H)</text>
  <text x="190" y="57" text-anchor="middle" font-size="8" fill="#388e3c">gaussian_blur_h</text>
  <rect x="290" y="30" width="100" height="35" rx="4" fill="#f3e5f5" stroke="#7b1fa2"/>
  <text x="340" y="45" text-anchor="middle" font-size="9" fill="#4a148c">Intermediate</text>
  <text x="340" y="57" text-anchor="middle" font-size="8" fill="#7b1fa2">R32Float R|W</text>
  <rect x="430" y="30" width="120" height="35" rx="4" fill="#e8f5e9" stroke="#388e3c"/>
  <text x="490" y="45" text-anchor="middle" font-size="9" fill="#1b5e20">Pass 2 (V)</text>
  <text x="490" y="57" text-anchor="middle" font-size="8" fill="#388e3c">gaussian_blur_v</text>
  <rect x="590" y="30" width="80" height="35" rx="4" fill="#fff3e0" stroke="#e65100"/>
  <text x="630" y="51" text-anchor="middle" font-size="9" fill="#bf360c">Output</text>
  <line x1="90" y1="47" x2="128" y2="47" stroke="#333" marker-end="url(#a7)"/>
  <line x1="250" y1="47" x2="288" y2="47" stroke="#333" marker-end="url(#a7)"/>
  <line x1="390" y1="47" x2="428" y2="47" stroke="#333" marker-end="url(#a7)"/>
  <line x1="550" y1="47" x2="588" y2="47" stroke="#333" marker-end="url(#a7)"/>
</svg>

**Texture-to-texture kernel catalog:**

| Kernel | Struct | Shader(s) | Passes | Notes |
|---|---|---|---|---|
| Gaussian | `GaussianBlur` | `gaussian_blur_h`, `gaussian_blur_v` | 2 | Separable, O(r) vs O(r²) |
| Sobel | `SobelFilter` | `sobel_3x3`, `gradient_magnitude` | 2 | Outputs: gx, gy, magnitude, direction |
| Canny | `CannyDetector` | Sobel + `canny_hysteresis` | 3 | Composes Sobel internally |
| Threshold | `ThresholdFilter` | `threshold_binary` / `adaptive` / `otsu` | 1–2 | Otsu needs histogram first |
| Color | `ColorConverter` | `rgba_to_gray`, `gray_to_rgba`, `rgba_to_hsv`, `hsv_to_rgba` | 1 | Per-pixel, no neighbors |
| Morphology | `MorphFilter` | `morph_erode`, `morph_dilate` | 1–2 | Open = erode+dilate |
| Pyramid | `ImagePyramidBuilder` | `pyramid_downsample` | N | 4 levels = 3 downsamples |
| Resize | `ResizeFilter` | `bilinear_resize` | 1 | Grid = output dimensions |
| Warp | `WarpFilter` | `warp_affine`, `warp_perspective` | 1 | Inverse transform per pixel |
| Bilateral | `BilateralFilter` | `bilateral_filter` | 1 | O(r²) — not separable |
| Dense Flow | `DenseFlowComputer` | `dense_flow` | 1 | Horn-Schunck |
| Connected | `ConnectedComponents` | `ccl_*` | iterative | Label propagation until convergence |
| Distance | `DistanceTransform` | `jfa_seed`, `jfa_step`, `jfa_distance` | 2+N | Jump Flooding, O(log n) passes |
| Template | `TemplateMatcher` | `template_match_ncc` | 1 | NCC score map |
| Integral | `IntegralImage` | `integral_*` | multi | Prefix sum (row then column) |

### 6.3 Hybrid Kernels

Consume textures, produce buffer output — or both.

**KLT Optical Flow** (`vx-vision/src/kernels/klt.rs`): Binds 8 textures (4-level pyramid × 2 frames) at indices 0–7, plus 3 buffers (prev_points, curr_points, status). Dispatch is 1D: one thread per point. Each thread does iterative Lucas-Kanade at each pyramid level (coarse-to-fine), reading texture neighborhoods and writing final position + tracking status to buffers.

**DoG (Difference-of-Gaussians):** Two-phase. First phase subtracts adjacent scale-space textures (texture→texture). Second phase finds 3D extrema across scale and space, appending `DoGKeypoint` to a buffer via atomic counter (texture→buffer).

### 6.4 Utility Kernels

**IndirectDispatch** (`vx-vision/src/kernels/indirect.rs`): Solves the "CPU round-trip" problem in FAST→Harris chaining.

FAST produces N corners (unknown until GPU completes). Without indirect dispatch: commit FAST → wait → read count → dispatch Harris with count. Two command buffers, one CPU stall.

With indirect dispatch: `prepare_indirect_args` reads the atomic counter and computes `MTLDispatchThreadgroupsIndirectArguments` on the GPU. Harris then dispatches using `dispatchThreadgroupsWithIndirectBuffer` — the GPU reads the thread count from the args buffer directly. Everything stays in one command buffer, zero CPU round-trips.

**Implementation note:** objc2-metal doesn't bind `dispatchThreadgroupsWithIndirectBuffer`. The code uses raw `msg_send!` to call it.

### 6.5 3D Reconstruction Kernels

Feature-gated: `#[cfg(feature = "reconstruction")]`.

| Category | Kernels | Pattern |
|---|---|---|
| Depth processing | `DepthFilter` (bilateral + median), `DepthInpaint` (JFA hole-fill), `DepthColorize` | texture→texture |
| Stereo | `SGMStereo` (Semi-Global Matching) | texture→texture (disparity map) |
| Point cloud | `DepthToCloud` (unprojection), `OutlierFilter`, `VoxelDownsample`, `NormalEstimation` | texture/buffer→buffer |
| Volumetric | `TSDFIntegrate`, `TSDFRaycast`, `MarchingCubes` | buffer→buffer (3D voxel grids) |
| Geometry | `Triangulate` | buffer→buffer |

Same patterns as core kernels. TSDF uses 3D buffer indexing (`res_x * res_y * res_z` voxels). MarchingCubes outputs triangle meshes to buffers with atomic vertex/index counters.

---

## 7. `#[repr(C)]` Contract

`vx-vision/src/types.rs`

Every params struct is passed to the GPU via `setBytes_length_atIndex` — a raw memcpy into the argument buffer. The GPU interprets the bytes according to the MSL struct layout. If the Rust and MSL layouts differ by even one byte, fields shift — silent data corruption, no error.

### Rules

1. **`#[repr(C)]`** — C-compatible field ordering. Without this, Rust may reorder fields for alignment optimization.
2. **`Pod + Zeroable`** (bytemuck) — certifies the type is safe to transmute from raw bytes.
3. **Field order matches MSL exactly.** Same names, same order, same types.
4. **Type mapping:**

| Rust | MSL | Size | Alignment | Notes |
|---|---|---|---|---|
| `u32` | `uint` | 4 | 4 | |
| `i32` | `int` | 4 | 4 | |
| `f32` | `float` | 4 | 4 | |
| `[f32; 2]` | `float2` | 8 | 8 | |
| `[f32; 3]` + `_pad: f32` | `float3` | 16 | **16** | Rust `[f32;3]` is only 4-byte aligned |
| `[f32; 4]` | `float4` | 16 | 16 | |
| `[u8; 4]` | `uchar4` | 4 | 4 | |

### float3 Alignment

MSL `float3` occupies 16 bytes (12 data + 4 padding) and requires 16-byte alignment. Rust's `[f32; 3]` is 12 bytes with 4-byte alignment. Without explicit padding, every field after `[f32; 3]` is offset by 4 bytes from what the GPU expects.

```rust
// Rust — matches MSL float3 layout
pub struct GpuPoint3D {
    pub position: [f32; 3],  // 12 bytes at offset 0
    pub _pad0: f32,          // 4 bytes → total 16, matching float3's footprint
    pub color: [u8; 4],      // offset 16
    pub normal: [f32; 3],    // 12 bytes at offset 20
    pub _pad1: f32,          // 4 bytes → total 16
}
```

```metal
struct GpuPoint3D {
    float3 position;  // 16 bytes (implicit padding to align)
    uchar4 color;     // 4 bytes
    float3 normal;    // 16 bytes
};
```

---

## 8. Dispatch Model

### 2D Per-Pixel

Image filters and per-pixel detectors. Grid = image dimensions.

```rust
let tew = pipeline.threadExecutionWidth();            // SIMD width: 32 on Apple Silicon
let max_tg = pipeline.maxTotalThreadsPerThreadgroup(); // typically 1024
let tg_h = (max_tg / tew).max(1);                     // 1024/32 = 32

let grid = MTLSize { width: w as usize, height: h as usize, depth: 1 };
let tg   = MTLSize { width: tew, height: tg_h, depth: 1 };  // 32x32 = 1024 threads
```

`threadExecutionWidth` is queried rather than hardcoded because future Apple Silicon may change the SIMD width — the query returns the hardware truth. The height is derived as `max_tg / tew` to maximize threadgroup occupancy: a 32x32 threadgroup fills 1024 threads (the hardware max). Smaller threadgroups waste SIMD lanes at threadgroup boundaries.

### 1D Per-Element

Buffer-output kernels processing variable-length arrays.

```rust
let grid = MTLSize { width: n_elements, height: 1, depth: 1 };
let tg   = MTLSize { width: tew, height: 1, depth: 1 };
```

### Why `dispatchThreads` (Non-Uniform)

`dispatchThreadgroups` requires grid dimensions to be a multiple of threadgroup size — manual ceil-division, easy to get wrong.

`dispatchThreads` takes the exact thread count. Metal internally handles partial threadgroups. Every shader still bounds-checks (`if gid.x >= width return`) because threads in the partial edge threadgroup may exceed the intended grid.

<svg viewBox="0 0 520 160" xmlns="http://www.w3.org/2000/svg" style="max-width:520px;font-family:monospace">
  <text x="10" y="16" font-size="11" font-weight="bold" fill="#333">2D Dispatch: 640x480, threadgroup 32x32</text>
  <rect x="10" y="25" width="200" height="100" fill="none" stroke="#1565c0" stroke-width="1.5"/>
  <text x="110" y="40" text-anchor="middle" font-size="9" fill="#1565c0">Grid: 640 x 480 threads</text>
  <rect x="15" y="45" width="24" height="24" fill="#e8f5e9" stroke="#388e3c" stroke-width="0.5"/>
  <rect x="41" y="45" width="24" height="24" fill="#e8f5e9" stroke="#388e3c" stroke-width="0.5"/>
  <rect x="67" y="45" width="24" height="24" fill="#e8f5e9" stroke="#388e3c" stroke-width="0.5"/>
  <text x="108" y="61" font-size="8" fill="#666">...</text>
  <rect x="15" y="71" width="24" height="24" fill="#e8f5e9" stroke="#388e3c" stroke-width="0.5"/>
  <rect x="41" y="71" width="24" height="24" fill="#e8f5e9" stroke="#388e3c" stroke-width="0.5"/>
  <rect x="67" y="71" width="24" height="24" fill="#e8f5e9" stroke="#388e3c" stroke-width="0.5"/>
  <text x="108" y="87" font-size="8" fill="#666">...</text>
  <!-- Partial groups at edge -->
  <rect x="170" y="45" width="24" height="24" fill="#fff3e0" stroke="#e65100" stroke-width="0.5"/>
  <rect x="170" y="71" width="24" height="24" fill="#fff3e0" stroke="#e65100" stroke-width="0.5"/>
  <!-- Legend -->
  <rect x="240" y="45" width="12" height="12" fill="#e8f5e9" stroke="#388e3c"/>
  <text x="258" y="55" font-size="9" fill="#333">Full 32x32 threadgroup</text>
  <rect x="240" y="65" width="12" height="12" fill="#fff3e0" stroke="#e65100"/>
  <text x="258" y="75" font-size="9" fill="#333">Partial (edge threads)</text>
  <text x="240" y="100" font-size="8" fill="#666">640 / 32 = 20 full columns</text>
  <text x="240" y="112" font-size="8" fill="#666">480 / 32 = 15 full rows</text>
  <text x="240" y="124" font-size="8" fill="#666">Edge threads: gid &gt;= dims → early return</text>
</svg>

**Performance: threadgroup sizing and occupancy.** Apple Silicon GPUs execute threads in SIMD groups (wavefronts) of 32. A 32x32 threadgroup = 1024 threads = 32 SIMD groups. The GPU schedules multiple threadgroups per compute unit to hide memory latency. Smaller threadgroups reduce this parallelism. Querying `maxTotalThreadsPerThreadgroup` and filling it completely maximizes occupancy.

---

## 9. Build System

`vx-vision/build.rs`

### Shader Compilation Pipeline

<svg viewBox="0 0 740 110" xmlns="http://www.w3.org/2000/svg" style="max-width:740px;font-family:monospace">
  <defs>
    <marker id="a8" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#333"/>
    </marker>
  </defs>
  <rect x="5" y="15" width="120" height="70" rx="5" fill="#e3f2fd" stroke="#1565c0"/>
  <text x="65" y="36" text-anchor="middle" font-size="9" font-weight="bold" fill="#0d47a1">shaders/</text>
  <text x="65" y="50" text-anchor="middle" font-size="8" fill="#1565c0">FastDetect.metal</text>
  <text x="65" y="62" text-anchor="middle" font-size="8" fill="#1565c0">GaussianBlur.metal</text>
  <text x="65" y="74" text-anchor="middle" font-size="8" fill="#1565c0">... (42 files)</text>
  <rect x="160" y="30" width="120" height="40" rx="5" fill="#e8f5e9" stroke="#388e3c"/>
  <text x="220" y="48" text-anchor="middle" font-size="9" fill="#1b5e20">xcrun metal -c</text>
  <text x="220" y="60" text-anchor="middle" font-size="7" fill="#388e3c">-std=metal3.1 -O2</text>
  <rect x="315" y="30" width="90" height="40" rx="5" fill="#fff3e0" stroke="#e65100"/>
  <text x="360" y="48" text-anchor="middle" font-size="9" fill="#bf360c">.air files</text>
  <text x="360" y="60" text-anchor="middle" font-size="8" fill="#e65100">(42 objects)</text>
  <rect x="440" y="30" width="110" height="40" rx="5" fill="#e8f5e9" stroke="#388e3c"/>
  <text x="495" y="54" text-anchor="middle" font-size="9" fill="#1b5e20">xcrun metallib</text>
  <rect x="585" y="30" width="100" height="40" rx="5" fill="#f3e5f5" stroke="#7b1fa2"/>
  <text x="635" y="48" text-anchor="middle" font-size="9" fill="#4a148c">vx.metallib</text>
  <text x="635" y="60" text-anchor="middle" font-size="8" fill="#7b1fa2">OUT_DIR</text>
  <text x="635" y="90" text-anchor="middle" font-size="8" fill="#666">include_bytes!() → embedded in binary</text>
  <line x1="125" y1="50" x2="158" y2="50" stroke="#333" marker-end="url(#a8)"/>
  <line x1="280" y1="50" x2="313" y2="50" stroke="#333" marker-end="url(#a8)"/>
  <line x1="405" y1="50" x2="438" y2="50" stroke="#333" marker-end="url(#a8)"/>
  <line x1="550" y1="50" x2="583" y2="50" stroke="#333" marker-end="url(#a8)"/>
</svg>

1. `build.rs` globs `shaders/*.metal`
2. Each `.metal` → `.air` (Apple Intermediate Representation):
   ```
   xcrun -sdk macosx metal -c -target air64-apple-macos14.0 -std=metal3.1 -O2 file.metal -o file.air
   ```
3. All `.air` → linked `vx.metallib`:
   ```
   xcrun -sdk macosx metallib *.air -o vx.metallib
   ```
4. Embedded at compile time:
   ```rust
   static METALLIB_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/vx.metallib"));
   ```
5. At runtime: written to temp file → loaded via `newLibraryWithURL_error`. Metal has no "load from bytes" API — it requires a file URL. The temp file is deleted immediately after loading.

**Auto-discovery:** `cargo:rerun-if-changed=shaders` watches the directory. New `.metal` files are picked up automatically — no `build.rs` edits needed.

**Compiler flags:**
- `-target air64-apple-macos14.0` — AIR bytecode for macOS 14+
- `-std=metal3.1` — MSL 3.1 (mesh shaders, ray tracing intrinsics available)
- `-O2` — full optimization (loop unrolling, dead code elimination, register allocation)
- `-Wno-unused-variable` — suppress warnings from template code

---

## 10. Error Handling

`vx-vision/src/error.rs`

```rust
#[non_exhaustive]
pub enum Error {
    DeviceNotFound,              // MTLCreateSystemDefaultDevice() returned nil
    ShaderMissing(String),       // newFunctionWithName() returned nil — typo in kernel name
    PipelineCompile(String),     // newComputePipelineStateWithFunction_error failed — MSL error
    BufferAlloc { bytes: usize },// newBufferWithLength returned nil — out of GPU memory
    TextureSizeMismatch,         // dimension mismatch between input/output textures
    InvalidConfig(String),       // bad parameter (negative radius, zero dimensions, etc.)
    Gpu(String),                 // runtime: commandBuffer/encoder creation failed, execution error
}
```

`From<String>` maps string errors to `Error::Gpu`. This bridges `vx-core` (which returns `Result<T, String>`) into `vx-vision`'s typed error enum.

`#[non_exhaustive]` allows adding variants without breaking downstream match statements.

---

## 11. Common Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Stale data on CPU readback | Guard dropped before `waitUntilCompleted`, or never created | Guard before commit, drop after wait |
| Garbage pixel output | Format mismatch (R32Float where R8Unorm expected) | Match shader's `texture2d<float>` access to texture format |
| Silent write failure | Input texture used as output (ShaderRead only) | Use `intermediate_*` or `output_*` |
| Shifted struct fields | `[f32; 3]` without `_pad: f32` in repr(C) struct | Pad every `float3` to 16 bytes |
| All-zero results | Atomic counter not zeroed before dispatch | `count_buf.write(&[0u32])` |
| Use-after-free crash | `EncodedBuffers` dropped before GPU completes | Keep encoded state alive until after wait |
| "Missing shader function" | `ns_string!()` doesn't match `kernel void` name | Exact string match between Rust and MSL |
| Threadgroup too large | Hardcoded threadgroup exceeds hardware max | Query `maxTotalThreadsPerThreadgroup()` |
| Incorrect buffer binding | Index in `setBuffer_atIndex` mismatches `[[buffer(N)]]` | Match Rust indices to MSL attribute indices |

---

## Appendix A: Shader Inventory

42 Metal shader files in `vx-vision/shaders/`:

| File | Kernel Function(s) | Category |
|---|---|---|
| `FastDetect.metal` | `fast_detect` | Feature detection |
| `HarrisResponse.metal` | `harris_response` | Feature detection |
| `NMS.metal` | `nms_suppress` | Feature detection |
| `ORBDescriptor.metal` | `orb_compute` | Feature description |
| `BruteMatcher.metal` | `hamming_distance`, `extract_matches` | Feature matching |
| `StereoMatch.metal` | `stereo_match` | Stereo |
| `KLTTracker.metal` | `klt_track_forward` | Optical flow |
| `DenseFlow.metal` | `dense_flow` | Optical flow |
| `GaussianBlur.metal` | `gaussian_blur_h`, `gaussian_blur_v` | Filtering |
| `Sobel.metal` | `sobel_3x3`, `gradient_magnitude` | Edge detection |
| `Canny.metal` | `canny_hysteresis` | Edge detection |
| `Threshold.metal` | `threshold_binary`, `threshold_adaptive`, `threshold_otsu` | Segmentation |
| `Morphology.metal` | `morph_erode`, `morph_dilate` | Morphology |
| `BilateralFilter.metal` | `bilateral_filter` | Filtering |
| `ColorConvert.metal` | `rgba_to_gray`, `gray_to_rgba`, `rgba_to_hsv`, `hsv_to_rgba` | Color |
| `Pyramid.metal` | `pyramid_downsample` | Scale space |
| `Resize.metal` | `bilinear_resize` | Geometry |
| `Warp.metal` | `warp_affine`, `warp_perspective` | Geometry |
| `IntegralImage.metal` | `integral_*` | Analysis |
| `Histogram.metal` | `histogram_compute` | Analysis |
| `HoughLines.metal` | `hough_vote`, `hough_peaks` | Line detection |
| `TemplateMatch.metal` | `template_match_ncc` | Template matching |
| `ConnectedComponents.metal` | `ccl_*` | Labeling |
| `DistanceTransform.metal` | `jfa_seed`, `jfa_step`, `jfa_distance` | Distance |
| `Homography.metal` | `score_homography` | Geometry |
| `IndirectArgs.metal` | `prepare_indirect_args` | Utility |
| `DoG.metal` | `dog_subtract`, `dog_extrema` | Scale space |
| `undistort.metal` | `undistort` | Calibration |
| `SGMStereo.metal` | `sgm_*` | Stereo |
| `DepthFilter.metal` | `depth_bilateral`, `depth_median` | Depth |
| `DepthInpaint.metal` | `depth_inpaint_*` | Depth |
| `DepthColorize.metal` | `depth_colorize` | Visualization |
| `DepthToCloud.metal` | `depth_to_cloud` | 3D reconstruction |
| `NormalEstimation.metal` | `normal_estimation_*` | 3D reconstruction |
| `OutlierFilter.metal` | `outlier_filter_*` | 3D reconstruction |
| `VoxelDownsample.metal` | `voxel_downsample_*` | 3D reconstruction |
| `TSDFIntegrate.metal` | `tsdf_integrate` | Volumetric |
| `TSDFRaycast.metal` | `tsdf_raycast` | Volumetric |
| `MarchingCubes.metal` | `marching_cubes_*` | Mesh extraction |
| `Triangulate.metal` | `triangulate_*` | Geometry |
| `PointCloudRender.metal` | `point_cloud_*` | Visualization |
| `MeshRender.metal` | `mesh_*` | Visualization |
