//! Sobel gradient filter with magnitude and direction.

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
use crate::types::SobelParams;

/// Result of a Sobel gradient computation.
pub struct SobelResult {
    /// Horizontal gradient Ix (R32Float).
    pub grad_x: Texture,
    /// Vertical gradient Iy (R32Float).
    pub grad_y: Texture,
    /// Gradient magnitude sqrt(Ix^2 + Iy^2) (R32Float).
    pub magnitude: Texture,
    /// Gradient direction atan2(Iy, Ix) in radians (R32Float).
    pub direction: Texture,
}

/// Compiled Sobel pipeline. Reusable across frames.
pub struct SobelFilter {
    sobel_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    mag_pipeline:   Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl SobelFilter {
    pub fn new(ctx: &Context) -> Result<Self, String> {
        let sobel_name = objc2_foundation::ns_string!("sobel_3x3");
        let mag_name   = objc2_foundation::ns_string!("gradient_magnitude");

        let sobel_func = ctx.library().newFunctionWithName(sobel_name)
            .ok_or_else(|| "Missing kernel function 'sobel_3x3'".to_string())?;
        let mag_func = ctx.library().newFunctionWithName(mag_name)
            .ok_or_else(|| "Missing kernel function 'gradient_magnitude'".to_string())?;

        let sobel_pipeline = ctx.device()
            .newComputePipelineStateWithFunction_error(&sobel_func)
            .map_err(|e| format!("Failed to create sobel_3x3 pipeline: {e}"))?;
        let mag_pipeline = ctx.device()
            .newComputePipelineStateWithFunction_error(&mag_func)
            .map_err(|e| format!("Failed to create gradient_magnitude pipeline: {e}"))?;

        Ok(Self { sobel_pipeline, mag_pipeline })
    }

    /// Computes Sobel gradients, magnitude, and direction from a grayscale (R8Unorm) texture.
    pub fn compute(
        &self,
        ctx:   &Context,
        input: &Texture,
    ) -> Result<SobelResult, String> {
        let w = input.width();
        let h = input.height();

        let grad_x    = Texture::intermediate_r32float(ctx.device(), w, h)?;
        let grad_y    = Texture::intermediate_r32float(ctx.device(), w, h)?;
        let magnitude = Texture::intermediate_r32float(ctx.device(), w, h)?;
        let direction = Texture::intermediate_r32float(ctx.device(), w, h)?;

        let params = SobelParams { width: w, height: h };

        let cmd_buf = ctx.queue().commandBuffer()
            .ok_or("Failed to create command buffer")?;

        {
            let encoder = cmd_buf.computeCommandEncoder()
                .ok_or("Failed to create compute encoder")?;
            Self::encode_sobel(&self.sobel_pipeline, &encoder, input, &grad_x, &grad_y, &params, w, h);
            encoder.endEncoding();
        }

        {
            let encoder = cmd_buf.computeCommandEncoder()
                .ok_or("Failed to create compute encoder")?;
            Self::encode_magnitude(&self.mag_pipeline, &encoder, &grad_x, &grad_y, &magnitude, &direction, &params, w, h);
            encoder.endEncoding();
        }

        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        Ok(SobelResult { grad_x, grad_y, magnitude, direction })
    }

    /// Computes only Sobel gradients (Ix, Iy) without magnitude or direction.
    pub fn gradients_only(
        &self,
        ctx:   &Context,
        input: &Texture,
    ) -> Result<(Texture, Texture), String> {
        let w = input.width();
        let h = input.height();

        let grad_x = Texture::intermediate_r32float(ctx.device(), w, h)?;
        let grad_y = Texture::intermediate_r32float(ctx.device(), w, h)?;

        let params = SobelParams { width: w, height: h };

        let cmd_buf = ctx.queue().commandBuffer()
            .ok_or("Failed to create command buffer")?;
        let encoder = cmd_buf.computeCommandEncoder()
            .ok_or("Failed to create compute encoder")?;

        Self::encode_sobel(&self.sobel_pipeline, &encoder, input, &grad_x, &grad_y, &params, w, h);

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        Ok((grad_x, grad_y))
    }

    fn encode_sobel(
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        encoder:  &ProtocolObject<dyn MTLComputeCommandEncoder>,
        input:    &Texture,
        grad_x:   &Texture,
        grad_y:   &Texture,
        params:   &SobelParams,
        width:    u32,
        height:   u32,
    ) {
        unsafe {
            encoder.setComputePipelineState(pipeline);
            encoder.setTexture_atIndex(Some(input.raw()),  0);
            encoder.setTexture_atIndex(Some(grad_x.raw()), 1);
            encoder.setTexture_atIndex(Some(grad_y.raw()), 2);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(params as *const SobelParams as *mut c_void),
                mem::size_of::<SobelParams>(),
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

    fn encode_magnitude(
        pipeline:  &ProtocolObject<dyn MTLComputePipelineState>,
        encoder:   &ProtocolObject<dyn MTLComputeCommandEncoder>,
        grad_x:    &Texture,
        grad_y:    &Texture,
        magnitude: &Texture,
        direction: &Texture,
        params:    &SobelParams,
        width:     u32,
        height:    u32,
    ) {
        unsafe {
            encoder.setComputePipelineState(pipeline);
            encoder.setTexture_atIndex(Some(grad_x.raw()),    0);
            encoder.setTexture_atIndex(Some(grad_y.raw()),    1);
            encoder.setTexture_atIndex(Some(magnitude.raw()), 2);
            encoder.setTexture_atIndex(Some(direction.raw()), 3);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(params as *const SobelParams as *mut c_void),
                mem::size_of::<SobelParams>(),
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
