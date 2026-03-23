//! Affine and perspective warp transforms via inverse mapping with bilinear sampling.

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
use crate::error::{Error, Result};
use crate::texture::Texture;
use crate::types::{WarpAffineParams, WarpPerspectiveParams};

/// Compiled warp pipelines.
pub struct ImageWarp {
    affine_pipeline:      Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    perspective_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

unsafe impl Send for ImageWarp {}
unsafe impl Sync for ImageWarp {}

impl ImageWarp {
    pub fn new(ctx: &Context) -> Result<Self> {
        let aff_name  = objc2_foundation::ns_string!("warp_affine");
        let persp_name = objc2_foundation::ns_string!("warp_perspective");

        let aff_func = ctx.library().newFunctionWithName(aff_name)
            .ok_or(Error::ShaderMissing("warp_affine".into()))?;
        let persp_func = ctx.library().newFunctionWithName(persp_name)
            .ok_or(Error::ShaderMissing("warp_perspective".into()))?;

        let affine_pipeline = ctx.device()
            .newComputePipelineStateWithFunction_error(&aff_func)
            .map_err(|e| Error::PipelineCompile(format!("warp_affine: {e}")))?;
        let perspective_pipeline = ctx.device()
            .newComputePipelineStateWithFunction_error(&persp_func)
            .map_err(|e| Error::PipelineCompile(format!("warp_perspective: {e}")))?;

        Ok(Self { affine_pipeline, perspective_pipeline })
    }

    /// Inverse affine warp (2x3 row-major). Out-of-bounds pixels are black.
    pub fn affine(
        &self,
        ctx:        &Context,
        input:      &Texture,
        output:     &Texture,
        inv_matrix: &[f32; 6],
    ) -> Result<()> {
        let params = WarpAffineParams {
            width:      output.width(),
            height:     output.height(),
            src_width:  input.width(),
            src_height: input.height(),
            m00: inv_matrix[0], m01: inv_matrix[1], m02: inv_matrix[2],
            m10: inv_matrix[3], m11: inv_matrix[4], m12: inv_matrix[5],
        };

        let cmd_buf = ctx.queue().commandBuffer()
            .ok_or(Error::Gpu("failed to create command buffer".into()))?;
        let encoder = cmd_buf.computeCommandEncoder()
            .ok_or(Error::Gpu("failed to create compute encoder".into()))?;

        Self::encode_2d(&self.affine_pipeline, &encoder, input, output, &params as *const _ as *const c_void, mem::size_of::<WarpAffineParams>());

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    /// Inverse perspective warp (3x3 row-major). Out-of-bounds pixels are black.
    pub fn perspective(
        &self,
        ctx:        &Context,
        input:      &Texture,
        output:     &Texture,
        inv_matrix: &[f32; 9],
    ) -> Result<()> {
        let params = WarpPerspectiveParams {
            width:      output.width(),
            height:     output.height(),
            src_width:  input.width(),
            src_height: input.height(),
            m00: inv_matrix[0], m01: inv_matrix[1], m02: inv_matrix[2],
            m10: inv_matrix[3], m11: inv_matrix[4], m12: inv_matrix[5],
            m20: inv_matrix[6], m21: inv_matrix[7], m22: inv_matrix[8],
        };

        let cmd_buf = ctx.queue().commandBuffer()
            .ok_or(Error::Gpu("failed to create command buffer".into()))?;
        let encoder = cmd_buf.computeCommandEncoder()
            .ok_or(Error::Gpu("failed to create compute encoder".into()))?;

        Self::encode_2d(&self.perspective_pipeline, &encoder, input, output, &params as *const _ as *const c_void, mem::size_of::<WarpPerspectiveParams>());

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    /// Encodes an affine warp without committing.
    pub fn encode_affine(
        &self,
        cmd_buf:    &ProtocolObject<dyn MTLCommandBuffer>,
        input:      &Texture,
        output:     &Texture,
        inv_matrix: &[f32; 6],
    ) -> Result<()> {
        let params = WarpAffineParams {
            width:      output.width(),
            height:     output.height(),
            src_width:  input.width(),
            src_height: input.height(),
            m00: inv_matrix[0], m01: inv_matrix[1], m02: inv_matrix[2],
            m10: inv_matrix[3], m11: inv_matrix[4], m12: inv_matrix[5],
        };

        let encoder = cmd_buf.computeCommandEncoder()
            .ok_or(Error::Gpu("failed to create compute encoder".into()))?;

        Self::encode_2d(&self.affine_pipeline, &encoder, input, output, &params as *const _ as *const c_void, mem::size_of::<WarpAffineParams>());

        encoder.endEncoding();
        Ok(())
    }

    /// Encodes a perspective warp without committing.
    pub fn encode_perspective(
        &self,
        cmd_buf:    &ProtocolObject<dyn MTLCommandBuffer>,
        input:      &Texture,
        output:     &Texture,
        inv_matrix: &[f32; 9],
    ) -> Result<()> {
        let params = WarpPerspectiveParams {
            width:      output.width(),
            height:     output.height(),
            src_width:  input.width(),
            src_height: input.height(),
            m00: inv_matrix[0], m01: inv_matrix[1], m02: inv_matrix[2],
            m10: inv_matrix[3], m11: inv_matrix[4], m12: inv_matrix[5],
            m20: inv_matrix[6], m21: inv_matrix[7], m22: inv_matrix[8],
        };

        let encoder = cmd_buf.computeCommandEncoder()
            .ok_or(Error::Gpu("failed to create compute encoder".into()))?;

        Self::encode_2d(&self.perspective_pipeline, &encoder, input, output, &params as *const _ as *const c_void, mem::size_of::<WarpPerspectiveParams>());

        encoder.endEncoding();
        Ok(())
    }

    fn encode_2d(
        pipeline:   &ProtocolObject<dyn MTLComputePipelineState>,
        encoder:    &ProtocolObject<dyn MTLComputeCommandEncoder>,
        input:      &Texture,
        output:     &Texture,
        params_ptr: *const c_void,
        params_len: usize,
    ) {
        let w = output.width();
        let h = output.height();

        unsafe {
            encoder.setComputePipelineState(pipeline);
            encoder.setTexture_atIndex(Some(input.raw()),  0);
            encoder.setTexture_atIndex(Some(output.raw()), 1);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(params_ptr as *mut c_void),
                params_len,
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
