//! Separable Gaussian blur (two-pass: horizontal then vertical).

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

/// Configuration for the Gaussian blur kernel.
#[derive(Clone, Debug)]
pub struct GaussianConfig {
    /// Standard deviation in pixels. Typical: 0.5--3.0.
    pub sigma: f32,

    /// Half-width of the convolution kernel. Full kernel spans `2 * radius + 1` taps.
    pub radius: u32,
}

impl Default for GaussianConfig {
    fn default() -> Self {
        Self { sigma: 1.0, radius: 3 }
    }
}

/// Compiled separable Gaussian blur pipeline. Reusable across frames.
pub struct GaussianBlur {
    h_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    v_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl GaussianBlur {
    /// Creates both horizontal and vertical pipelines from the context's shader library.
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

    /// Blurs `input` into `output` using a separable Gaussian kernel.
    ///
    /// Both textures must have identical dimensions. An intermediate R32Float
    /// scratch texture is allocated internally.
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

        // Horizontal pass writes here; vertical pass reads it
        let intermediate = Texture::intermediate_r32float(ctx.device(), width, height)?;

        let params = GaussianParams {
            width,
            height,
            sigma:  config.sigma,
            radius: config.radius,
        };

        let cmd_buf = ctx.queue().commandBuffer()
            .ok_or("Failed to create command buffer")?;

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

        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        Ok(())
    }

    /// Encodes both blur passes into an existing command buffer without committing.
    ///
    /// The intermediate texture is returned inside [`GaussianEncodedState`]
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

    fn encode_pass(
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        encoder:  &ProtocolObject<dyn MTLComputeCommandEncoder>,
        src:      &Texture,
        dst:      &Texture,
        params:   &GaussianParams,
        width:    u32,
        height:   u32,
    ) {
        // SAFETY: setBytes requires a valid pointer; encoder ops interact with device state.
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

/// Intermediate state returned by [`GaussianBlur::encode`]. Keeps the scratch texture alive.
pub struct GaussianEncodedState {
    pub _intermediate: Texture,
}
