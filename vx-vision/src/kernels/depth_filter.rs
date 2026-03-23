//! Depth map post-processing filters: bilateral smoothing and median filter.

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
use crate::types::{DepthBilateralParams, DepthMedianParams};

/// Configuration for the depth bilateral filter.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct DepthFilterConfig {
    /// Spatial window half-width in pixels.
    pub radius: i32,
    /// Spatial Gaussian sigma.
    pub sigma_spatial: f32,
    /// Depth Gaussian sigma (in depth units). Controls edge preservation.
    pub sigma_depth: f32,
}

impl Default for DepthFilterConfig {
    fn default() -> Self {
        Self {
            radius: 3,
            sigma_spatial: 5.0,
            sigma_depth: 0.05,
        }
    }
}

impl DepthFilterConfig {
    pub fn new(radius: i32, sigma_spatial: f32, sigma_depth: f32) -> Self {
        Self {
            radius,
            sigma_spatial,
            sigma_depth,
        }
    }
}

/// Configuration for the depth median filter.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct DepthMedianConfig {
    /// Filter radius (1 = 3x3, 2 = 5x5).
    pub radius: i32,
}

impl Default for DepthMedianConfig {
    fn default() -> Self {
        Self { radius: 1 }
    }
}

/// Compiled depth filter pipelines.
pub struct DepthFilter {
    bilateral_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    median_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

unsafe impl Send for DepthFilter {}
unsafe impl Sync for DepthFilter {}

impl DepthFilter {
    pub fn new(ctx: &Context) -> Result<Self> {
        let bilateral_func = ctx
            .library()
            .newFunctionWithName(objc2_foundation::ns_string!("depth_bilateral"))
            .ok_or(Error::ShaderMissing("depth_bilateral".into()))?;
        let median_func = ctx
            .library()
            .newFunctionWithName(objc2_foundation::ns_string!("depth_median"))
            .ok_or(Error::ShaderMissing("depth_median".into()))?;

        let bilateral_pipeline = ctx
            .device()
            .newComputePipelineStateWithFunction_error(&bilateral_func)
            .map_err(|e| Error::PipelineCompile(format!("depth_bilateral: {e}")))?;
        let median_pipeline = ctx
            .device()
            .newComputePipelineStateWithFunction_error(&median_func)
            .map_err(|e| Error::PipelineCompile(format!("depth_median: {e}")))?;

        Ok(Self {
            bilateral_pipeline,
            median_pipeline,
        })
    }

    /// Applies depth bilateral filter. Synchronous.
    pub fn apply_bilateral(
        &self,
        ctx: &Context,
        input: &Texture,
        output: &Texture,
        config: &DepthFilterConfig,
    ) -> Result<()> {
        let w = input.width();
        let h = input.height();
        let params = DepthBilateralParams {
            width: w,
            height: h,
            radius: config.radius,
            sigma_spatial: config.sigma_spatial,
            sigma_depth: config.sigma_depth,
        };

        let cmd_buf = ctx
            .queue()
            .commandBuffer()
            .ok_or(Error::Gpu("failed to create command buffer".into()))?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or(Error::Gpu("failed to create compute encoder".into()))?;

        Self::encode_bilateral_pass(
            &self.bilateral_pipeline,
            &encoder,
            input,
            output,
            &params,
            w,
            h,
        );

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    /// Applies depth median filter. Synchronous.
    pub fn apply_median(
        &self,
        ctx: &Context,
        input: &Texture,
        output: &Texture,
        config: &DepthMedianConfig,
    ) -> Result<()> {
        let w = input.width();
        let h = input.height();
        let params = DepthMedianParams {
            width: w,
            height: h,
            radius: config.radius,
        };

        let cmd_buf = ctx
            .queue()
            .commandBuffer()
            .ok_or(Error::Gpu("failed to create command buffer".into()))?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or(Error::Gpu("failed to create compute encoder".into()))?;

        Self::encode_median_pass(
            &self.median_pipeline,
            &encoder,
            input,
            output,
            &params,
            w,
            h,
        );

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    /// Encodes bilateral filter into a command buffer without committing.
    pub fn encode_bilateral(
        &self,
        cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
        input: &Texture,
        output: &Texture,
        config: &DepthFilterConfig,
    ) -> Result<()> {
        let w = input.width();
        let h = input.height();
        let params = DepthBilateralParams {
            width: w,
            height: h,
            radius: config.radius,
            sigma_spatial: config.sigma_spatial,
            sigma_depth: config.sigma_depth,
        };

        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or(Error::Gpu("failed to create compute encoder".into()))?;
        Self::encode_bilateral_pass(
            &self.bilateral_pipeline,
            &encoder,
            input,
            output,
            &params,
            w,
            h,
        );
        encoder.endEncoding();
        Ok(())
    }

    /// Encodes median filter into a command buffer without committing.
    pub fn encode_median(
        &self,
        cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
        input: &Texture,
        output: &Texture,
        config: &DepthMedianConfig,
    ) -> Result<()> {
        let w = input.width();
        let h = input.height();
        let params = DepthMedianParams {
            width: w,
            height: h,
            radius: config.radius,
        };

        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or(Error::Gpu("failed to create compute encoder".into()))?;
        Self::encode_median_pass(
            &self.median_pipeline,
            &encoder,
            input,
            output,
            &params,
            w,
            h,
        );
        encoder.endEncoding();
        Ok(())
    }

    fn encode_bilateral_pass(
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        input: &Texture,
        output: &Texture,
        params: &DepthBilateralParams,
        w: u32,
        h: u32,
    ) {
        unsafe {
            encoder.setComputePipelineState(pipeline);
            encoder.setTexture_atIndex(Some(input.raw()), 0);
            encoder.setTexture_atIndex(Some(output.raw()), 1);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(params as *const DepthBilateralParams as *mut c_void),
                mem::size_of::<DepthBilateralParams>(),
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

    fn encode_median_pass(
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        input: &Texture,
        output: &Texture,
        params: &DepthMedianParams,
        w: u32,
        h: u32,
    ) {
        unsafe {
            encoder.setComputePipelineState(pipeline);
            encoder.setTexture_atIndex(Some(input.raw()), 0);
            encoder.setTexture_atIndex(Some(output.raw()), 1);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(params as *const DepthMedianParams as *mut c_void),
                mem::size_of::<DepthMedianParams>(),
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
