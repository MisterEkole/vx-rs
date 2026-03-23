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
use crate::error::{Error, Result};
use crate::texture::Texture;
use crate::types::IntegralParams;

/// Keeps the intermediate texture alive until the command buffer completes.
pub struct IntegralEncodedState {
    _intermediate: Texture,
    /// The output integral image texture.
    pub output: Texture,
}

/// Compiled integral image pipeline.
pub struct IntegralImage {
    rows_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    cols_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

unsafe impl Send for IntegralImage {}
unsafe impl Sync for IntegralImage {}

impl IntegralImage {
    pub fn new(ctx: &Context) -> Result<Self> {
        let rows_name = objc2_foundation::ns_string!("integral_rows");
        let cols_name = objc2_foundation::ns_string!("integral_cols");

        let rows_func = ctx.library().newFunctionWithName(rows_name)
            .ok_or(Error::ShaderMissing("integral_rows".into()))?;
        let cols_func = ctx.library().newFunctionWithName(cols_name)
            .ok_or(Error::ShaderMissing("integral_cols".into()))?;

        let rows_pipeline = ctx.device()
            .newComputePipelineStateWithFunction_error(&rows_func)
            .map_err(|e| Error::PipelineCompile(format!("integral_rows: {e}")))?;
        let cols_pipeline = ctx.device()
            .newComputePipelineStateWithFunction_error(&cols_func)
            .map_err(|e| Error::PipelineCompile(format!("integral_cols: {e}")))?;

        Ok(Self { rows_pipeline, cols_pipeline })
    }

    /// Computes the integral image. Returns an R32Float texture.
    pub fn compute(
        &self,
        ctx:   &Context,
        input: &Texture,
    ) -> Result<Texture> {
        let w = input.width();
        let h = input.height();

        let intermediate = Texture::intermediate_r32float(ctx.device(), w, h)?;
        let output       = Texture::intermediate_r32float(ctx.device(), w, h)?;

        let params = IntegralParams { width: w, height: h };

        let cmd_buf = ctx.queue().commandBuffer()
            .ok_or(Error::Gpu("failed to create command buffer".into()))?;

        {
            let encoder = cmd_buf.computeCommandEncoder()
                .ok_or(Error::Gpu("failed to create compute encoder".into()))?;
            Self::encode_1d(&self.rows_pipeline, &encoder, input, &intermediate, &params, h as usize);
            encoder.endEncoding();
        }

        {
            let encoder = cmd_buf.computeCommandEncoder()
                .ok_or(Error::Gpu("failed to create compute encoder".into()))?;
            Self::encode_1d(&self.cols_pipeline, &encoder, &intermediate, &output, &params, w as usize);
            encoder.endEncoding();
        }

        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        Ok(output)
    }

    /// Encodes both passes without committing.
    pub fn encode(
        &self,
        ctx:     &Context,
        cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
        input:   &Texture,
    ) -> Result<IntegralEncodedState> {
        let w = input.width();
        let h = input.height();

        let intermediate = Texture::intermediate_r32float(ctx.device(), w, h)?;
        let output       = Texture::intermediate_r32float(ctx.device(), w, h)?;

        let params = IntegralParams { width: w, height: h };

        {
            let encoder = cmd_buf.computeCommandEncoder()
                .ok_or(Error::Gpu("failed to create compute encoder".into()))?;
            Self::encode_1d(&self.rows_pipeline, &encoder, input, &intermediate, &params, h as usize);
            encoder.endEncoding();
        }

        {
            let encoder = cmd_buf.computeCommandEncoder()
                .ok_or(Error::Gpu("failed to create compute encoder".into()))?;
            Self::encode_1d(&self.cols_pipeline, &encoder, &intermediate, &output, &params, w as usize);
            encoder.endEncoding();
        }

        Ok(IntegralEncodedState { _intermediate: intermediate, output })
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
