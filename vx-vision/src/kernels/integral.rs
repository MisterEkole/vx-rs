//! Summed area table (integral image) via two-pass GPU prefix sums (rows then columns).

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
use crate::types::IntegralParams;

/// Compiled integral image pipeline.
pub struct IntegralImage {
    rows_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    cols_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl IntegralImage {
    pub fn new(ctx: &Context) -> Result<Self, String> {
        let rows_name = objc2_foundation::ns_string!("integral_rows");
        let cols_name = objc2_foundation::ns_string!("integral_cols");

        let rows_func = ctx.library().newFunctionWithName(rows_name)
            .ok_or_else(|| "Missing kernel function 'integral_rows'".to_string())?;
        let cols_func = ctx.library().newFunctionWithName(cols_name)
            .ok_or_else(|| "Missing kernel function 'integral_cols'".to_string())?;

        let rows_pipeline = ctx.device()
            .newComputePipelineStateWithFunction_error(&rows_func)
            .map_err(|e| format!("Failed to create integral_rows pipeline: {e}"))?;
        let cols_pipeline = ctx.device()
            .newComputePipelineStateWithFunction_error(&cols_func)
            .map_err(|e| format!("Failed to create integral_cols pipeline: {e}"))?;

        Ok(Self { rows_pipeline, cols_pipeline })
    }

    /// Computes the integral image of a grayscale input.
    ///
    /// Returns an R32Float texture.
    pub fn compute(
        &self,
        ctx:   &Context,
        input: &Texture,
    ) -> Result<Texture, String> {
        let w = input.width();
        let h = input.height();

        let intermediate = Texture::intermediate_r32float(ctx.device(), w, h)?;
        let output       = Texture::intermediate_r32float(ctx.device(), w, h)?;

        let params = IntegralParams { width: w, height: h };

        let cmd_buf = ctx.queue().commandBuffer()
            .ok_or("Failed to create command buffer")?;

        // Pass 1: prefix sum per row
        {
            let encoder = cmd_buf.computeCommandEncoder()
                .ok_or("Failed to create compute encoder")?;
            Self::encode_1d(&self.rows_pipeline, &encoder, input, &intermediate, &params, h as usize);
            encoder.endEncoding();
        }

        // Pass 2: prefix sum per column
        {
            let encoder = cmd_buf.computeCommandEncoder()
                .ok_or("Failed to create compute encoder")?;
            Self::encode_1d(&self.cols_pipeline, &encoder, &intermediate, &output, &params, w as usize);
            encoder.endEncoding();
        }

        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        Ok(output)
    }

    fn encode_1d(
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        encoder:  &ProtocolObject<dyn MTLComputeCommandEncoder>,
        input:    &Texture,
        output:   &Texture,
        params:   &IntegralParams,
        n:        usize,
    ) {
        unsafe {
            encoder.setComputePipelineState(pipeline);
            encoder.setTexture_atIndex(Some(input.raw()),  0);
            encoder.setTexture_atIndex(Some(output.raw()), 1);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(params as *const IntegralParams as *mut c_void),
                mem::size_of::<IntegralParams>(),
                0,
            );

            let tew     = pipeline.threadExecutionWidth();
            let grid    = MTLSize { width: n,   height: 1, depth: 1 };
            let tg_size = MTLSize { width: tew, height: 1, depth: 1 };
            encoder.dispatchThreads_threadsPerThreadgroup(grid, tg_size);
        }
    }
}
