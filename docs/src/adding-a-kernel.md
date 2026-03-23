# Adding a Kernel

Five steps to add a new GPU kernel to VX.

## 1. Write the Metal shader

Create `vx-vision/shaders/YourKernel.metal`:

```metal
#include <metal_stdlib>
using namespace metal;

struct YourParams {
    uint width;
    uint height;
    float some_param;
};

kernel void your_kernel(
    texture2d<float, access::read>  input  [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    constant YourParams& params            [[buffer(0)]],
    uint2 gid                              [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) return;

    float4 pixel = input.read(gid);
    // ... your computation ...
    output.write(result, gid);
}
```

The build system auto-discovers `.metal` files — no `build.rs` changes needed.

## 2. Add the parameter struct

In `vx-vision/src/types.rs`:

```rust
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct YourParams {
    pub width: u32,
    pub height: u32,
    pub some_param: f32,
}
```

Must match the Metal struct field-by-field. Same types, same order, same padding. See the [Architecture](architecture.md) page for type mapping.

## 3. Write the Rust kernel

Create `vx-vision/src/kernels/your_kernel.rs`:

```rust
use crate::context::Context;
use crate::error::{Error, Result};
use crate::texture::Texture;
use crate::types::YourParams;
// ... Metal imports ...

#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct YourConfig {
    pub some_param: f32,
}

impl Default for YourConfig {
    fn default() -> Self {
        Self { some_param: 1.0 }
    }
}

pub struct YourKernel {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl YourKernel {
    pub fn new(ctx: &Context) -> Result<Self> {
        let name = objc2_foundation::ns_string!("your_kernel");
        let func = ctx.library().newFunctionWithName(name)
            .ok_or(Error::ShaderMissing("your_kernel".into()))?;
        let pipeline = ctx.device()
            .newComputePipelineStateWithFunction_error(&func)
            .map_err(|e| Error::PipelineCompile(format!("your_kernel: {e}")))?;
        Ok(Self { pipeline })
    }

    /// Sync method: creates command buffer, dispatches, waits.
    pub fn apply(
        &self, ctx: &Context, input: &Texture, output: &Texture, config: &YourConfig,
    ) -> Result<()> {
        let cmd_buf = ctx.queue().commandBuffer()
            .ok_or(Error::Gpu("failed to create command buffer".into()))?;
        self.encode_pass(&cmd_buf, input, output, config)?;
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    /// Pipeline encoding: writes into existing command buffer.
    pub fn encode(
        &self, cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
        input: &Texture, output: &Texture, config: &YourConfig,
    ) -> Result<()> {
        self.encode_pass(cmd_buf, input, output, config)
    }

    fn encode_pass(/* ... */) -> Result<()> {
        // set pipeline, textures, params, dispatch
    }
}

unsafe impl Send for YourKernel {}
unsafe impl Sync for YourKernel {}
```

## 4. Register the module

In `vx-vision/src/kernels/mod.rs`:

```rust
pub mod your_kernel;
```

## 5. Add tests

In `vx-vision/tests/test_kernels.rs`, add a test that creates a synthetic image, runs the kernel, and verifies output properties.

## Checklist

- [ ] Metal shader compiles (check `cargo build` output)
- [ ] `#[repr(C)]` struct matches MSL struct exactly
- [ ] Kernel has both sync method and `encode()` for pipelining
- [ ] Config struct has `Default`, `#[non_exhaustive]`, `Clone`, `Debug`
- [ ] `Send + Sync` implemented on kernel struct
- [ ] Module registered in `mod.rs`
- [ ] Test passes
