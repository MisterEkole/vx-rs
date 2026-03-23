//! Binary, adaptive, and Otsu thresholding.

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
use crate::kernels::histogram::Histogram;
use crate::kernels::integral::IntegralImage;
use crate::texture::Texture;
use crate::types::{ThresholdParams, AdaptiveThresholdParams};

/// Configuration for adaptive thresholding.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct AdaptiveThresholdConfig {
    /// Half-window size for the local mean computation.
    pub radius: i32,
    /// Constant subtracted from the local mean before comparison.
    pub c: f32,
    /// Invert the result.
    pub invert: bool,
}

impl AdaptiveThresholdConfig {
    /// Creates a config with the given parameters.
    pub fn new(radius: i32, c: f32, invert: bool) -> Self {
        Self { radius, c, invert }
    }
}

impl Default for AdaptiveThresholdConfig {
    fn default() -> Self {
        Self { radius: 7, c: 0.02, invert: false }
    }
}

/// Compiled thresholding pipelines.
pub struct Threshold {
    binary_pipeline:   Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    adaptive_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    histogram:         Histogram,
    integral:          IntegralImage,
}

impl Threshold {
    pub fn new(ctx: &Context) -> Result<Self> {
        let bin_name  = objc2_foundation::ns_string!("threshold_binary");
        let adpt_name = objc2_foundation::ns_string!("threshold_adaptive");

        let bin_func = ctx.library().newFunctionWithName(bin_name)
            .ok_or(Error::ShaderMissing("threshold_binary".into()))?;
        let adpt_func = ctx.library().newFunctionWithName(adpt_name)
            .ok_or(Error::ShaderMissing("threshold_adaptive".into()))?;

        let binary_pipeline = ctx.device()
            .newComputePipelineStateWithFunction_error(&bin_func)
            .map_err(|e| Error::PipelineCompile(format!("threshold_binary: {e}")))?;
        let adaptive_pipeline = ctx.device()
            .newComputePipelineStateWithFunction_error(&adpt_func)
            .map_err(|e| Error::PipelineCompile(format!("threshold_adaptive: {e}")))?;

        let histogram = Histogram::new(ctx)?;
        let integral  = IntegralImage::new(ctx)?;

        Ok(Self { binary_pipeline, adaptive_pipeline, histogram, integral })
    }

    /// Global binary threshold. Pixels above `threshold` become 1.0, below become 0.0.
    pub fn binary(
        &self,
        ctx:       &Context,
        input:     &Texture,
        output:    &Texture,
        threshold: f32,
        invert:    bool,
    ) -> Result<()> {
        let w = input.width();
        let h = input.height();
        let params = ThresholdParams {
            width:     w,
            height:    h,
            threshold,
            invert:    if invert { 1 } else { 0 },
        };

        let cmd_buf = ctx.queue().commandBuffer()
            .ok_or(Error::Gpu("failed to create command buffer".into()))?;
        let encoder = cmd_buf.computeCommandEncoder()
            .ok_or(Error::Gpu("failed to create compute encoder".into()))?;

        Self::encode_binary_pass(&self.binary_pipeline, &encoder, input, output, &params, w, h);

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    /// Encodes a binary threshold without committing.
    pub fn encode_binary(
        &self,
        cmd_buf:   &ProtocolObject<dyn MTLCommandBuffer>,
        input:     &Texture,
        output:    &Texture,
        threshold: f32,
        invert:    bool,
    ) -> Result<()> {
        let w = input.width();
        let h = input.height();
        let params = ThresholdParams {
            width: w, height: h, threshold,
            invert: if invert { 1 } else { 0 },
        };

        let encoder = cmd_buf.computeCommandEncoder()
            .ok_or(Error::Gpu("failed to create compute encoder".into()))?;
        Self::encode_binary_pass(&self.binary_pipeline, &encoder, input, output, &params, w, h);
        encoder.endEncoding();
        Ok(())
    }

    /// Encodes adaptive threshold without committing. Requires a precomputed integral image.
    pub fn encode_adaptive(
        &self,
        cmd_buf:  &ProtocolObject<dyn MTLCommandBuffer>,
        input:    &Texture,
        integral: &Texture,
        output:   &Texture,
        config:   &AdaptiveThresholdConfig,
    ) -> Result<()> {
        let w = input.width();
        let h = input.height();
        let params = AdaptiveThresholdParams {
            width:  w,
            height: h,
            radius: config.radius,
            c:      config.c,
            invert: if config.invert { 1 } else { 0 },
        };

        let encoder = cmd_buf.computeCommandEncoder()
            .ok_or(Error::Gpu("failed to create compute encoder".into()))?;

        unsafe {
            encoder.setComputePipelineState(&self.adaptive_pipeline);
            encoder.setTexture_atIndex(Some(input.raw()),    0);
            encoder.setTexture_atIndex(Some(integral.raw()), 1);
            encoder.setTexture_atIndex(Some(output.raw()),   2);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(&params as *const AdaptiveThresholdParams as *mut c_void),
                mem::size_of::<AdaptiveThresholdParams>(),
                0,
            );

            let tew    = self.adaptive_pipeline.threadExecutionWidth();
            let max_tg = self.adaptive_pipeline.maxTotalThreadsPerThreadgroup();
            let tg_h   = (max_tg / tew).max(1);
            let grid    = MTLSize { width: w as usize, height: h as usize, depth: 1 };
            let tg_size = MTLSize { width: tew,        height: tg_h,       depth: 1 };
            encoder.dispatchThreads_threadsPerThreadgroup(grid, tg_size);
        }

        encoder.endEncoding();
        Ok(())
    }

    /// Adaptive threshold using a precomputed integral image.
    pub fn adaptive(
        &self,
        ctx:      &Context,
        input:    &Texture,
        integral: &Texture,
        output:   &Texture,
        config:   &AdaptiveThresholdConfig,
    ) -> Result<()> {
        let w = input.width();
        let h = input.height();
        let params = AdaptiveThresholdParams {
            width:  w,
            height: h,
            radius: config.radius,
            c:      config.c,
            invert: if config.invert { 1 } else { 0 },
        };

        let cmd_buf = ctx.queue().commandBuffer()
            .ok_or(Error::Gpu("failed to create command buffer".into()))?;
        let encoder = cmd_buf.computeCommandEncoder()
            .ok_or(Error::Gpu("failed to create compute encoder".into()))?;

        unsafe {
            encoder.setComputePipelineState(&self.adaptive_pipeline);
            encoder.setTexture_atIndex(Some(input.raw()),    0);
            encoder.setTexture_atIndex(Some(integral.raw()), 1);
            encoder.setTexture_atIndex(Some(output.raw()),   2);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(&params as *const AdaptiveThresholdParams as *mut c_void),
                mem::size_of::<AdaptiveThresholdParams>(),
                0,
            );

            let tew    = self.adaptive_pipeline.threadExecutionWidth();
            let max_tg = self.adaptive_pipeline.maxTotalThreadsPerThreadgroup();
            let tg_h   = (max_tg / tew).max(1);
            let grid    = MTLSize { width: w as usize, height: h as usize, depth: 1 };
            let tg_size = MTLSize { width: tew,        height: tg_h,       depth: 1 };
            encoder.dispatchThreads_threadsPerThreadgroup(grid, tg_size);
        }

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    /// Otsu's method: auto-selects threshold, returns the computed value.
    pub fn otsu(
        &self,
        ctx:    &Context,
        input:  &Texture,
        output: &Texture,
    ) -> Result<f32> {
        let hist = self.histogram.compute(ctx, input)?;
        let total_pixels = (input.width() as f64) * (input.height() as f64);
        let threshold = otsu_threshold(&hist, total_pixels);
        self.binary(ctx, input, output, threshold, false)?;
        Ok(threshold)
    }

    /// Adaptive threshold that computes the integral image internally.
    pub fn adaptive_auto(
        &self,
        ctx:    &Context,
        input:  &Texture,
        output: &Texture,
        config: &AdaptiveThresholdConfig,
    ) -> Result<()> {
        let integral = self.integral.compute(ctx, input)?;
        self.adaptive(ctx, input, &integral, output, config)
    }

    fn encode_binary_pass(
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        encoder:  &ProtocolObject<dyn MTLComputeCommandEncoder>,
        input:    &Texture,
        output:   &Texture,
        params:   &ThresholdParams,
        width:    u32,
        height:   u32,
    ) {
        unsafe {
            encoder.setComputePipelineState(pipeline);
            encoder.setTexture_atIndex(Some(input.raw()),  0);
            encoder.setTexture_atIndex(Some(output.raw()), 1);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(params as *const ThresholdParams as *mut c_void),
                mem::size_of::<ThresholdParams>(),
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

unsafe impl Send for Threshold {}
unsafe impl Sync for Threshold {}

fn otsu_threshold(hist: &[u32; 256], total_pixels: f64) -> f32 {
    let mut sum_total = 0.0f64;
    for (i, &count) in hist.iter().enumerate() {
        sum_total += (i as f64) * (count as f64);
    }

    let mut sum_bg = 0.0f64;
    let mut weight_bg = 0.0f64;
    let mut max_variance = 0.0f64;
    let mut best_threshold = 0usize;

    for (t, &count) in hist.iter().enumerate() {
        weight_bg += count as f64;
        if weight_bg == 0.0 { continue; }

        let weight_fg = total_pixels - weight_bg;
        if weight_fg == 0.0 { break; }

        sum_bg += (t as f64) * (count as f64);

        let mean_bg = sum_bg / weight_bg;
        let mean_fg = (sum_total - sum_bg) / weight_fg;

        let variance = weight_bg * weight_fg * (mean_bg - mean_fg) * (mean_bg - mean_fg);

        if variance > max_variance {
            max_variance = variance;
            best_threshold = t;
        }
    }

    best_threshold as f32 / 255.0
}
