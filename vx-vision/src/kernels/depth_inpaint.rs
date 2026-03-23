//! Depth map hole filling via iterative nearest-neighbor inpainting.

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
use crate::types::DepthInpaintParams;

/// Configuration for depth inpainting.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct DepthInpaintConfig {
    /// Maximum number of fill iterations. Each doubles the fill radius.
    pub max_iterations: u32,
}

impl Default for DepthInpaintConfig {
    fn default() -> Self {
        Self { max_iterations: 8 }
    }
}

/// Compiled depth inpainting pipeline.
pub struct DepthInpaint {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

unsafe impl Send for DepthInpaint {}
unsafe impl Sync for DepthInpaint {}

impl DepthInpaint {
    pub fn new(ctx: &Context) -> Result<Self> {
        let func = ctx
            .library()
            .newFunctionWithName(objc2_foundation::ns_string!("depth_inpaint"))
            .ok_or(Error::ShaderMissing("depth_inpaint".into()))?;
        let pipeline = ctx
            .device()
            .newComputePipelineStateWithFunction_error(&func)
            .map_err(|e| Error::PipelineCompile(format!("depth_inpaint: {e}")))?;
        Ok(Self { pipeline })
    }

    /// Fills holes in a depth map. Synchronous.
    pub fn apply(
        &self,
        ctx: &Context,
        input: &Texture,
        output: &Texture,
        config: &DepthInpaintConfig,
    ) -> Result<()> {
        let w = input.width();
        let h = input.height();

        // Create a second intermediate for ping-pong
        let intermediate = Texture::intermediate_r32float(ctx.device(), w, h)?;

        let mut step = 1i32;
        for i in 0..config.max_iterations {
            let (src, dst) = if i % 2 == 0 {
                if i == 0 {
                    (input, &intermediate)
                } else {
                    (output as &Texture, &intermediate)
                }
            } else {
                (&intermediate as &Texture, output)
            };

            let params = DepthInpaintParams {
                width: w,
                height: h,
                step_size: step,
            };

            let cmd_buf = ctx
                .queue()
                .commandBuffer()
                .ok_or(Error::Gpu("failed to create command buffer".into()))?;
            let encoder = cmd_buf
                .computeCommandEncoder()
                .ok_or(Error::Gpu("failed to create compute encoder".into()))?;

            Self::encode_pass(&self.pipeline, &encoder, src, dst, &params, w, h);

            encoder.endEncoding();
            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();

            step *= 2;
        }

        // Odd iteration count leaves result in intermediate; copy to output
        if config.max_iterations % 2 == 1 {
            let params = DepthInpaintParams {
                width: w,
                height: h,
                step_size: step,
            };
            let cmd_buf = ctx
                .queue()
                .commandBuffer()
                .ok_or(Error::Gpu("failed to create command buffer".into()))?;
            let encoder = cmd_buf
                .computeCommandEncoder()
                .ok_or(Error::Gpu("failed to create compute encoder".into()))?;
            Self::encode_pass(
                &self.pipeline,
                &encoder,
                &intermediate,
                output,
                &params,
                w,
                h,
            );
            encoder.endEncoding();
            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();
        }

        Ok(())
    }

    fn encode_pass(
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        input: &Texture,
        output: &Texture,
        params: &DepthInpaintParams,
        w: u32,
        h: u32,
    ) {
        unsafe {
            encoder.setComputePipelineState(pipeline);
            encoder.setTexture_atIndex(Some(input.raw()), 0);
            encoder.setTexture_atIndex(Some(output.raw()), 1);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(params as *const DepthInpaintParams as *mut c_void),
                mem::size_of::<DepthInpaintParams>(),
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
