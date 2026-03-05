// vx-vision/src/kernels/gaussian.rs
//
// Rust binding for the separable Gaussian blur kernel (GaussianBlur.metal).
//
// Two-pass pipeline:
//   Pass 1 (gaussian_blur_h): horizontal 1D convolution → R32Float intermediate
//   Pass 2 (gaussian_blur_v): vertical   1D convolution → R8Unorm output
//
// Both passes run in a single command buffer. An internally allocated
// R32Float scratch texture (ShaderRead | ShaderWrite) carries the result
// of the horizontal pass to the vertical pass.
//
// Kernel weights are computed from the Gaussian formula inside the shader.
// `radius` is the half-width of the kernel: the full kernel spans
// (2 * radius + 1) taps. Rule of thumb: radius ≥ ceil(3 * sigma).
//
// Usage:
//   let ctx = Context::new()?;
//   let blur = GaussianBlur::new(&ctx)?;
//   let output = ctx.texture_output_gray8(width, height)?;
//   blur.apply(&ctx, &input, &output, &GaussianConfig::default())?;
//   let pixels = output.read_gray8();

use core::ffi::c_void;
use core::ptr::NonNull;
use std::mem;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice,
    MTLLibrary, MTLSize,
};

use crate::context::Context;
use crate::texture::Texture;
use crate::types::GaussianParams;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// User-facing config for the Gaussian blur kernel.
#[derive(Clone, Debug)]
pub struct GaussianConfig {
    /// Standard deviation of the Gaussian in pixels.
    /// Controls the amount of blur. Typical: 0.5–3.0.
    pub sigma: f32,

    /// Half-width of the convolution kernel in pixels.
    /// Full kernel spans `2 * radius + 1` taps.
    /// Rule of thumb: `radius = ceil(3.0 * sigma)` as usize.
    pub radius: u32,
}

impl Default for GaussianConfig {
    fn default() -> Self {
        Self { sigma: 1.0, radius: 3 }
    }
}

// ---------------------------------------------------------------------------
// Blur kernel
// ---------------------------------------------------------------------------

/// Compiled separable Gaussian blur pipeline. Create once, reuse across frames.
pub struct GaussianBlur {
    h_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    v_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl GaussianBlur {
    // --------------------------------------------------------------------
    // Construction
    // --------------------------------------------------------------------

    /// Build both compute pipelines from the context's shader library.
    pub fn new(ctx: &Context) -> Result<Self, String> {
        let h_name = objc2_foundation::ns_string!("gaussian_blur_h");
        let v_name = objc2_foundation::ns_string!("gaussian_blur_v");

        let h_func = ctx.library().newFunctionWithName(h_name)
            .ok_or_else(|| "Missing kernel function 'gaussian_blur_h'".to_string())?;
        let v_func = ctx.library().newFunctionWithName(v_name)
            .ok_or_else(|| "Missing kernel function 'gaussian_blur_v'".to_string())?;

        let h_pipeline = ctx.device()
            .newComputePipelineStateWithFunction_error(&h_func)
            .map_err(|e| format!("Failed to create gaussian_blur_h pipeline: {e}"))?;
        let v_pipeline = ctx.device()
            .newComputePipelineStateWithFunction_error(&v_func)
            .map_err(|e| format!("Failed to create gaussian_blur_v pipeline: {e}"))?;

        Ok(Self { h_pipeline, v_pipeline })
    }

    // --------------------------------------------------------------------
    // Synchronous dispatch
    // --------------------------------------------------------------------

    /// Blur `input` into `output` using a separable Gaussian kernel.
    ///
    /// - `input`  — grayscale source (R8Unorm, ShaderRead)
    /// - `output` — blurred destination (R8Unorm, ShaderWrite);
    ///              create with [`Context::texture_output_gray8`], read back
    ///              with [`Texture::read_gray8`]
    ///
    /// `input` and `output` must have identical dimensions.
    /// An intermediate R32Float scratch texture is allocated internally.
    ///
    /// Synchronous: encodes both passes, commits, waits for GPU completion.
    pub fn apply(
        &self,
        ctx:    &Context,
        input:  &Texture,
        output: &Texture,
        config: &GaussianConfig,
    ) -> Result<(), String> {
        let width  = input.width();
        let height = input.height();

        // Scratch texture: horizontal pass writes here, vertical pass reads it
        let intermediate = Texture::intermediate_r32float(ctx.device(), width, height)?;

        let params = GaussianParams {
            width,
            height,
            sigma:  config.sigma,
            radius: config.radius,
        };

        let cmd_buf = ctx.queue().commandBuffer()
            .ok_or("Failed to create command buffer")?;

        // Pass 1: horizontal blur  (input → intermediate)
        {
            let encoder = cmd_buf.computeCommandEncoder()
                .ok_or("Failed to create horizontal blur encoder")?;
            Self::encode_pass(&self.h_pipeline, &encoder, input, &intermediate, &params, width, height);
            encoder.endEncoding();
        }

        // Pass 2: vertical blur  (intermediate → output)
        {
            let encoder = cmd_buf.computeCommandEncoder()
                .ok_or("Failed to create vertical blur encoder")?;
            Self::encode_pass(&self.v_pipeline, &encoder, &intermediate, output, &params, width, height);
            encoder.endEncoding();
        }

        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        Ok(())
    }

    // --------------------------------------------------------------------
    // Encode-only (for pipelining)
    // --------------------------------------------------------------------

    /// Encode both blur passes into an existing command buffer without
    /// committing. Two new compute encoders are created and ended inside;
    /// they are appended after whatever encoders `cmd_buf` already has.
    ///
    /// The intermediate texture is returned inside `GaussianEncodedState`
    /// to keep it alive until the command buffer completes.
    pub fn encode(
        &self,
        ctx:     &Context,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        input:   &Texture,
        output:  &Texture,
        config:  &GaussianConfig,
    ) -> Result<GaussianEncodedState, String> {
        let width  = input.width();
        let height = input.height();

        let intermediate = Texture::intermediate_r32float(ctx.device(), width, height)?;

        let params = GaussianParams {
            width,
            height,
            sigma:  config.sigma,
            radius: config.radius,
        };

        {
            let encoder = cmd_buf.computeCommandEncoder()
                .ok_or("Failed to create horizontal blur encoder")?;
            Self::encode_pass(&self.h_pipeline, &encoder, input, &intermediate, &params, width, height);
            encoder.endEncoding();
        }

        {
            let encoder = cmd_buf.computeCommandEncoder()
                .ok_or("Failed to create vertical blur encoder")?;
            Self::encode_pass(&self.v_pipeline, &encoder, &intermediate, output, &params, width, height);
            encoder.endEncoding();
        }

        Ok(GaussianEncodedState { _intermediate: intermediate })
    }

    // --------------------------------------------------------------------
    // Internal
    // --------------------------------------------------------------------

    fn encode_pass(
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        encoder:  &ProtocolObject<dyn MTLComputeCommandEncoder>,
        src:      &Texture,
        dst:      &Texture,
        params:   &GaussianParams,
        width:    u32,
        height:   u32,
    ) {
        // SAFETY: GPU encoder operations interact with device state.
        //
        // Binding matches GaussianBlur.metal (both passes share the same layout):
        //   texture(0) = src    (read)
        //   texture(1) = dst    (write)
        //   buffer(0)  = params (GaussianParams, constant)
        //
        // 2D dispatch: one thread per pixel.
        unsafe {
            encoder.setComputePipelineState(pipeline);
            encoder.setTexture_atIndex(Some(src.raw()), 0);
            encoder.setTexture_atIndex(Some(dst.raw()), 1);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(params as *const GaussianParams as *mut c_void),
                mem::size_of::<GaussianParams>(),
                0,
            );

            let tew    = pipeline.threadExecutionWidth();
            let max_tg = pipeline.maxTotalThreadsPerThreadgroup();
            let tg_h   = (max_tg / tew).max(1);

            let grid    = MTLSize { width: width as usize,  height: height as usize, depth: 1 };
            let tg_size = MTLSize { width: tew,             height: tg_h,            depth: 1 };

            encoder.dispatchThreads_threadsPerThreadgroup(grid, tg_size);
        }
    }
}

/// Holds the intermediate scratch texture alive until the command buffer
/// completes. Returned by [`GaussianBlur::encode`].
pub struct GaussianEncodedState {
    pub _intermediate: Texture,
}
