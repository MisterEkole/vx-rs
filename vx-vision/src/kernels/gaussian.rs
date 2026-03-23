//! Separable Gaussian blur (two-pass: horizontal then vertical).

use core::ffi::c_void;
use core::ptr::NonNull;
use std::mem;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLLibrary, MTLSize,
};

use crate::context::Context;
use crate::error::{Error, Result};
use crate::texture::Texture;
use crate::types::GaussianParams;

/// Configuration for the Gaussian blur kernel.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct GaussianConfig {
    /// Standard deviation in pixels (typical: 0.5--3.0).
    pub sigma: f32,
    /// Half-width of the convolution kernel (`2 * radius + 1` taps).
    pub radius: u32,
}

impl Default for GaussianConfig {
    fn default() -> Self {
        Self {
            sigma: 1.0,
            radius: 3,
        }
    }
}

/// Compiled separable Gaussian blur pipeline.
pub struct GaussianBlur {
    h_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    v_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl GaussianBlur {
    /// Creates both horizontal and vertical blur pipelines.
    pub fn new(ctx: &Context) -> Result<Self> {
        let h_name = objc2_foundation::ns_string!("gaussian_blur_h");
        let v_name = objc2_foundation::ns_string!("gaussian_blur_v");

        let h_func = ctx
            .library()
            .newFunctionWithName(h_name)
            .ok_or(Error::ShaderMissing("gaussian_blur_h".into()))?;
        let v_func = ctx
            .library()
            .newFunctionWithName(v_name)
            .ok_or(Error::ShaderMissing("gaussian_blur_v".into()))?;

        let h_pipeline = ctx
            .device()
            .newComputePipelineStateWithFunction_error(&h_func)
            .map_err(|e| Error::PipelineCompile(format!("gaussian_blur_h: {e}")))?;
        let v_pipeline = ctx
            .device()
            .newComputePipelineStateWithFunction_error(&v_func)
            .map_err(|e| Error::PipelineCompile(format!("gaussian_blur_v: {e}")))?;

        Ok(Self {
            h_pipeline,
            v_pipeline,
        })
    }

    /// Blurs `input` into `output`. Synchronous.
    pub fn apply(
        &self,
        ctx: &Context,
        input: &Texture,
        output: &Texture,
        config: &GaussianConfig,
    ) -> Result<()> {
        let width = input.width();
        let height = input.height();

        let intermediate = Texture::intermediate_r32float(ctx.device(), width, height)?;

        let params = GaussianParams {
            width,
            height,
            sigma: config.sigma,
            radius: config.radius,
        };

        let cmd_buf = ctx
            .queue()
            .commandBuffer()
            .ok_or(Error::Gpu("failed to create command buffer".into()))?;

        {
            let encoder = cmd_buf
                .computeCommandEncoder()
                .ok_or(Error::Gpu("failed to create compute encoder".into()))?;
            Self::encode_pass(
                &self.h_pipeline,
                &encoder,
                input,
                &intermediate,
                &params,
                width,
                height,
            );
            encoder.endEncoding();
        }

        {
            let encoder = cmd_buf
                .computeCommandEncoder()
                .ok_or(Error::Gpu("failed to create compute encoder".into()))?;
            Self::encode_pass(
                &self.v_pipeline,
                &encoder,
                &intermediate,
                output,
                &params,
                width,
                height,
            );
            encoder.endEncoding();
        }

        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        Ok(())
    }

    /// Encodes both blur passes into `cmd_buf` without committing.
    pub fn encode(
        &self,
        ctx: &Context,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        input: &Texture,
        output: &Texture,
        config: &GaussianConfig,
    ) -> Result<GaussianEncodedState> {
        let width = input.width();
        let height = input.height();

        let intermediate = Texture::intermediate_r32float(ctx.device(), width, height)?;

        let params = GaussianParams {
            width,
            height,
            sigma: config.sigma,
            radius: config.radius,
        };

        {
            let encoder = cmd_buf
                .computeCommandEncoder()
                .ok_or(Error::Gpu("failed to create compute encoder".into()))?;
            Self::encode_pass(
                &self.h_pipeline,
                &encoder,
                input,
                &intermediate,
                &params,
                width,
                height,
            );
            encoder.endEncoding();
        }

        {
            let encoder = cmd_buf
                .computeCommandEncoder()
                .ok_or(Error::Gpu("failed to create compute encoder".into()))?;
            Self::encode_pass(
                &self.v_pipeline,
                &encoder,
                &intermediate,
                output,
                &params,
                width,
                height,
            );
            encoder.endEncoding();
        }

        Ok(GaussianEncodedState {
            _intermediate: intermediate,
        })
    }

    fn encode_pass(
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        src: &Texture,
        dst: &Texture,
        params: &GaussianParams,
        width: u32,
        height: u32,
    ) {
        unsafe {
            encoder.setComputePipelineState(pipeline);
            encoder.setTexture_atIndex(Some(src.raw()), 0);
            encoder.setTexture_atIndex(Some(dst.raw()), 1);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(params as *const GaussianParams as *mut c_void),
                mem::size_of::<GaussianParams>(),
                0,
            );

            let tew = pipeline.threadExecutionWidth();
            let max_tg = pipeline.maxTotalThreadsPerThreadgroup();
            let tg_h = (max_tg / tew).max(1);

            let grid = MTLSize {
                width: width as usize,
                height: height as usize,
                depth: 1,
            };
            let tg_size = MTLSize {
                width: tew,
                height: tg_h,
                depth: 1,
            };

            encoder.dispatchThreads_threadsPerThreadgroup(grid, tg_size);
        }
    }
}

unsafe impl Send for GaussianBlur {}
unsafe impl Sync for GaussianBlur {}

/// Keeps the intermediate texture alive until the command buffer completes.
pub struct GaussianEncodedState {
    pub _intermediate: Texture,
}
