//! GPU image resize with bilinear interpolation.

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
use crate::types::ResizeParams;

/// Compiled bilinear resize pipeline.
pub struct ImageResize {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl ImageResize {
    pub fn new(ctx: &Context) -> Result<Self, String> {
        let name = objc2_foundation::ns_string!("resize_bilinear");
        let func = ctx.library().newFunctionWithName(name)
            .ok_or_else(|| "Missing kernel function 'resize_bilinear'".to_string())?;
        let pipeline = ctx.device()
            .newComputePipelineStateWithFunction_error(&func)
            .map_err(|e| format!("Failed to create resize pipeline: {e}"))?;
        Ok(Self { pipeline })
    }

    /// Resize `input` into `output` using bilinear interpolation.
    /// Target dimensions are determined by `output`.
    pub fn apply(
        &self,
        ctx:    &Context,
        input:  &Texture,
        output: &Texture,
    ) -> Result<(), String> {
        let params = ResizeParams {
            src_width:  input.width(),
            src_height: input.height(),
            dst_width:  output.width(),
            dst_height: output.height(),
        };

        let cmd_buf = ctx.queue().commandBuffer()
            .ok_or("Failed to create command buffer")?;
        let encoder = cmd_buf.computeCommandEncoder()
            .ok_or("Failed to create compute encoder")?;

        Self::encode_into(&self.pipeline, &encoder, input, output, &params);

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    /// Encodes the resize into an existing compute encoder without committing.
    pub fn encode(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        input:   &Texture,
        output:  &Texture,
    ) {
        let params = ResizeParams {
            src_width:  input.width(),
            src_height: input.height(),
            dst_width:  output.width(),
            dst_height: output.height(),
        };
        Self::encode_into(&self.pipeline, encoder, input, output, &params);
    }

    fn encode_into(
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        encoder:  &ProtocolObject<dyn MTLComputeCommandEncoder>,
        input:    &Texture,
        output:   &Texture,
        params:   &ResizeParams,
    ) {
        let w = params.dst_width;
        let h = params.dst_height;

        unsafe {
            encoder.setComputePipelineState(pipeline);
            encoder.setTexture_atIndex(Some(input.raw()),  0);
            encoder.setTexture_atIndex(Some(output.raw()), 1);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(params as *const ResizeParams as *mut c_void),
                mem::size_of::<ResizeParams>(),
                0,
            );

            let tew    = pipeline.threadExecutionWidth();
            let max_tg = pipeline.maxTotalThreadsPerThreadgroup();
            let tg_h   = (max_tg / tew).max(1);
            let grid    = MTLSize { width: w as usize, height: h as usize, depth: 1 };
            let tg_size = MTLSize { width: tew,        height: tg_h,       depth: 1 };
            encoder.dispatchThreads_threadsPerThreadgroup(grid, tg_size);
        }
    }
}
