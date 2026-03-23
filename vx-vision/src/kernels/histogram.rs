//! GPU histogram computation and equalization.
//!
//! No `encode()` method is provided because histogram requires a CPU readback
//! of the 256-bin accumulation buffer between dispatch and any downstream use.

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
use crate::types::HistogramParams;
use vx_gpu::UnifiedBuffer;

/// Compiled histogram pipelines. Requires CPU readback between passes.
pub struct Histogram {
    compute_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    equalize_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl Histogram {
    pub fn new(ctx: &Context) -> Result<Self> {
        let comp_name = objc2_foundation::ns_string!("histogram_compute");
        let eq_name = objc2_foundation::ns_string!("histogram_equalize");

        let comp_func = ctx
            .library()
            .newFunctionWithName(comp_name)
            .ok_or(Error::ShaderMissing("histogram_compute".into()))?;
        let eq_func = ctx
            .library()
            .newFunctionWithName(eq_name)
            .ok_or(Error::ShaderMissing("histogram_equalize".into()))?;

        let compute_pipeline = ctx
            .device()
            .newComputePipelineStateWithFunction_error(&comp_func)
            .map_err(|e| Error::PipelineCompile(format!("histogram_compute: {e}")))?;
        let equalize_pipeline = ctx
            .device()
            .newComputePipelineStateWithFunction_error(&eq_func)
            .map_err(|e| Error::PipelineCompile(format!("histogram_equalize: {e}")))?;

        Ok(Self {
            compute_pipeline,
            equalize_pipeline,
        })
    }

    /// 256-bin histogram of a grayscale image.
    pub fn compute(&self, ctx: &Context, input: &Texture) -> Result<[u32; 256]> {
        let w = input.width();
        let h = input.height();
        let params = HistogramParams {
            width: w,
            height: h,
        };

        let mut bins_buf: UnifiedBuffer<u32> = UnifiedBuffer::new(ctx.device(), 256)?;
        bins_buf.write(&[0u32; 256]);

        let _guard = bins_buf.gpu_guard();

        let cmd_buf = ctx
            .queue()
            .commandBuffer()
            .ok_or(Error::Gpu("failed to create command buffer".into()))?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or(Error::Gpu("failed to create compute encoder".into()))?;

        unsafe {
            encoder.setComputePipelineState(&self.compute_pipeline);
            encoder.setTexture_atIndex(Some(input.raw()), 0);
            encoder.setBuffer_offset_atIndex(Some(bins_buf.metal_buffer()), 0, 0);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(&params as *const HistogramParams as *mut c_void),
                mem::size_of::<HistogramParams>(),
                1,
            );

            let tew = self.compute_pipeline.threadExecutionWidth();
            let max_tg = self.compute_pipeline.maxTotalThreadsPerThreadgroup();
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

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        drop(_guard);

        let mut result = [0u32; 256];
        result.copy_from_slice(&bins_buf.as_slice()[..256]);
        Ok(result)
    }

    /// Histogram equalization.
    pub fn equalize(&self, ctx: &Context, input: &Texture, output: &Texture) -> Result<()> {
        let w = input.width();
        let h = input.height();
        let total_pixels = (w as f32) * (h as f32);

        let hist = self.compute(ctx, input)?;

        let mut cdf = [0.0f32; 256];
        let mut cumsum = 0u32;
        let cdf_min = hist.iter().copied().find(|&v| v > 0).unwrap_or(0) as f32;

        for (i, &count) in hist.iter().enumerate() {
            cumsum += count;
            cdf[i] = ((cumsum as f32 - cdf_min) / (total_pixels - cdf_min)).clamp(0.0, 1.0);
        }

        let mut cdf_buf: UnifiedBuffer<f32> = UnifiedBuffer::new(ctx.device(), 256)?;
        cdf_buf.write(&cdf);

        let _guard = cdf_buf.gpu_guard();
        let params = HistogramParams {
            width: w,
            height: h,
        };

        let cmd_buf = ctx
            .queue()
            .commandBuffer()
            .ok_or(Error::Gpu("failed to create command buffer".into()))?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or(Error::Gpu("failed to create compute encoder".into()))?;

        unsafe {
            encoder.setComputePipelineState(&self.equalize_pipeline);
            encoder.setTexture_atIndex(Some(input.raw()), 0);
            encoder.setTexture_atIndex(Some(output.raw()), 1);
            encoder.setBuffer_offset_atIndex(Some(cdf_buf.metal_buffer()), 0, 0);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(&params as *const HistogramParams as *mut c_void),
                mem::size_of::<HistogramParams>(),
                1,
            );

            let tew = self.equalize_pipeline.threadExecutionWidth();
            let max_tg = self.equalize_pipeline.maxTotalThreadsPerThreadgroup();
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

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        drop(_guard);
        Ok(())
    }
}

unsafe impl Send for Histogram {}
unsafe impl Sync for Histogram {}
