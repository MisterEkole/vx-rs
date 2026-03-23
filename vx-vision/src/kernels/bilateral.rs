//! Edge-preserving bilateral filter.

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
use crate::types::BilateralParams;

/// Configuration for the bilateral filter.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct BilateralConfig {
    /// Spatial window half-width in pixels.
    pub radius: i32,
    /// Spatial Gaussian sigma.
    pub sigma_spatial: f32,
    /// Range Gaussian sigma. Smaller values preserve more edges.
    pub sigma_range: f32,
}

impl BilateralConfig {
    /// Creates a config with the given parameters.
    pub fn new(radius: i32, sigma_spatial: f32, sigma_range: f32) -> Self {
        Self {
            radius,
            sigma_spatial,
            sigma_range,
        }
    }
}

impl Default for BilateralConfig {
    fn default() -> Self {
        Self {
            radius: 5,
            sigma_spatial: 10.0,
            sigma_range: 0.1,
        }
    }
}

/// Compiled bilateral filter pipeline.
pub struct BilateralFilter {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

unsafe impl Send for BilateralFilter {}
unsafe impl Sync for BilateralFilter {}

impl BilateralFilter {
    pub fn new(ctx: &Context) -> Result<Self> {
        let name = objc2_foundation::ns_string!("bilateral_filter");
        let func = ctx
            .library()
            .newFunctionWithName(name)
            .ok_or(Error::ShaderMissing("bilateral_filter".into()))?;
        let pipeline = ctx
            .device()
            .newComputePipelineStateWithFunction_error(&func)
            .map_err(|e| Error::PipelineCompile(format!("bilateral_filter: {e}")))?;
        Ok(Self { pipeline })
    }

    /// Applies the bilateral filter. Synchronous.
    pub fn apply(
        &self,
        ctx: &Context,
        input: &Texture,
        output: &Texture,
        config: &BilateralConfig,
    ) -> Result<()> {
        let w = input.width();
        let h = input.height();
        let params = BilateralParams {
            width: w,
            height: h,
            radius: config.radius,
            sigma_spatial: config.sigma_spatial,
            sigma_range: config.sigma_range,
        };

        let cmd_buf = ctx
            .queue()
            .commandBuffer()
            .ok_or(Error::Gpu("failed to create command buffer".into()))?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or(Error::Gpu("failed to create compute encoder".into()))?;

        Self::encode_pass(&self.pipeline, &encoder, input, output, &params, w, h);

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    /// Encodes the bilateral filter without committing.
    pub fn encode(
        &self,
        cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
        input: &Texture,
        output: &Texture,
        config: &BilateralConfig,
    ) -> Result<()> {
        let w = input.width();
        let h = input.height();
        let params = BilateralParams {
            width: w,
            height: h,
            radius: config.radius,
            sigma_spatial: config.sigma_spatial,
            sigma_range: config.sigma_range,
        };

        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or(Error::Gpu("failed to create compute encoder".into()))?;

        Self::encode_pass(&self.pipeline, &encoder, input, output, &params, w, h);

        encoder.endEncoding();
        Ok(())
    }

    fn encode_pass(
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        input: &Texture,
        output: &Texture,
        params: &BilateralParams,
        w: u32,
        h: u32,
    ) {
        unsafe {
            encoder.setComputePipelineState(pipeline);
            encoder.setTexture_atIndex(Some(input.raw()), 0);
            encoder.setTexture_atIndex(Some(output.raw()), 1);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(params as *const BilateralParams as *mut c_void),
                mem::size_of::<BilateralParams>(),
                0,
            );

            let tew = pipeline.threadExecutionWidth();
            let max_tg = pipeline.maxTotalThreadsPerThreadgroup();
            let tg_h = (max_tg / tew).max(1);
            let grid = MTLSize {
                width: w as usize,
                height: h as usize,
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
