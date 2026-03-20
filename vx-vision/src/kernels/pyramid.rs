//! Gaussian image pyramid via 2x downsample with a 5-tap filter.

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
use crate::types::PyramidParams;

/// Compiled GPU pyramid pipeline.
pub struct PyramidBuilder {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl PyramidBuilder {
    pub fn new(ctx: &Context) -> Result<Self, String> {
        let name = objc2_foundation::ns_string!("pyramid_downsample");
        let func = ctx.library().newFunctionWithName(name)
            .ok_or_else(|| "Missing kernel function 'pyramid_downsample'".to_string())?;
        let pipeline = ctx.device()
            .newComputePipelineStateWithFunction_error(&func)
            .map_err(|e| format!("Failed to create pyramid pipeline: {e}"))?;
        Ok(Self { pipeline })
    }

    /// Builds `n_levels` pyramid levels (level 0 = input, not returned).
    ///
    /// Returns textures for levels `1..n_levels`, each half the prior resolution.
    pub fn build(
        &self,
        ctx:      &Context,
        input:    &Texture,
        n_levels: usize,
    ) -> Result<Vec<Texture>, String> {
        if n_levels <= 1 {
            return Ok(Vec::new());
        }

        let cmd_buf = ctx.queue().commandBuffer()
            .ok_or("Failed to create command buffer")?;

        let mut src_w = input.width();
        let mut src_h = input.height();
        let mut prev_tex: &Texture = input;
        let mut levels: Vec<Texture> = Vec::with_capacity(n_levels - 1);

        for _ in 1..n_levels {
            let dst_w = (src_w + 1) / 2;
            let dst_h = (src_h + 1) / 2;

            let output = Texture::output_gray8(ctx.device(), dst_w, dst_h)?;

            let encoder = cmd_buf.computeCommandEncoder()
                .ok_or("Failed to create compute encoder")?;

            let params = PyramidParams {
                src_width:  src_w,
                src_height: src_h,
                dst_width:  dst_w,
                dst_height: dst_h,
            };

            Self::encode_pass(&self.pipeline, &encoder, prev_tex, &output, &params, dst_w, dst_h);
            encoder.endEncoding();

            levels.push(output);
            src_w = dst_w;
            src_h = dst_h;
            prev_tex = levels.last().unwrap();
        }

        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        Ok(levels)
    }

    /// Downsample a single level to half resolution.
    pub fn downsample(
        &self,
        ctx:    &Context,
        input:  &Texture,
        output: &Texture,
    ) -> Result<(), String> {
        let params = PyramidParams {
            src_width:  input.width(),
            src_height: input.height(),
            dst_width:  output.width(),
            dst_height: output.height(),
        };

        let cmd_buf = ctx.queue().commandBuffer()
            .ok_or("Failed to create command buffer")?;
        let encoder = cmd_buf.computeCommandEncoder()
            .ok_or("Failed to create compute encoder")?;

        Self::encode_pass(&self.pipeline, &encoder, input, output, &params, output.width(), output.height());

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    fn encode_pass(
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        encoder:  &ProtocolObject<dyn MTLComputeCommandEncoder>,
        src:      &Texture,
        dst:      &Texture,
        params:   &PyramidParams,
        width:    u32,
        height:   u32,
    ) {
        unsafe {
            encoder.setComputePipelineState(pipeline);
            encoder.setTexture_atIndex(Some(src.raw()), 0);
            encoder.setTexture_atIndex(Some(dst.raw()), 1);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(params as *const PyramidParams as *mut c_void),
                mem::size_of::<PyramidParams>(),
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
