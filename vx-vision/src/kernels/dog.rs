//! Difference-of-Gaussians scale-space keypoint detection.

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
use crate::kernels::gaussian::{GaussianBlur, GaussianConfig};
use crate::texture::Texture;
use crate::types::{DoGExtremaParams, DoGKeypoint, DoGParams};

/// Configuration for DoG keypoint detection.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct DoGConfig {
    /// Scale levels per octave (typically 3--5).
    pub n_levels: usize,
    /// Base sigma for the initial blur.
    pub base_sigma: f32,
    /// Scale ratio between adjacent levels.
    pub scale_factor: f32,
    /// Minimum |DoG| response for keypoint acceptance.
    pub contrast_threshold: f32,
    /// Maximum number of keypoints returned.
    pub max_keypoints: u32,
}

impl Default for DoGConfig {
    fn default() -> Self {
        Self {
            n_levels: 4,
            base_sigma: 1.6,
            scale_factor: 1.2599, // 2^(1/3)
            contrast_threshold: 0.04,
            max_keypoints: 4096,
        }
    }
}

/// DoG detection compute pipelines.
pub struct DoGDetector {
    subtract_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    extrema_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl DoGDetector {
    pub fn new(ctx: &Context) -> Result<Self> {
        let sub_name = objc2_foundation::ns_string!("dog_subtract");
        let ext_name = objc2_foundation::ns_string!("dog_extrema");

        let sub_func = ctx
            .library()
            .newFunctionWithName(sub_name)
            .ok_or_else(|| Error::ShaderMissing("dog_subtract".into()))?;
        let ext_func = ctx
            .library()
            .newFunctionWithName(ext_name)
            .ok_or_else(|| Error::ShaderMissing("dog_extrema".into()))?;

        let subtract_pipeline = ctx
            .device()
            .newComputePipelineStateWithFunction_error(&sub_func)
            .map_err(|e| Error::PipelineCompile(format!("dog_subtract: {e}")))?;
        let extrema_pipeline = ctx
            .device()
            .newComputePipelineStateWithFunction_error(&ext_func)
            .map_err(|e| Error::PipelineCompile(format!("dog_extrema: {e}")))?;

        Ok(Self {
            subtract_pipeline,
            extrema_pipeline,
        })
    }

    /// Detects scale-space keypoints across all DoG levels.
    pub fn detect(
        &self,
        ctx: &Context,
        blur: &GaussianBlur,
        input: &Texture,
        config: &DoGConfig,
    ) -> Result<Vec<DoGKeypoint>> {
        let w = input.width();
        let h = input.height();

        let n_blur = config.n_levels + 1;
        let mut blurred: Vec<Texture> = Vec::with_capacity(n_blur);

        let b0 = Texture::intermediate_r32float(ctx.device(), w, h)?;
        let sigma0 = config.base_sigma;
        let radius0 = (sigma0 * 3.0).ceil() as u32;
        blur.apply(
            ctx,
            input,
            &b0,
            &GaussianConfig {
                sigma: sigma0,
                radius: radius0,
            },
        )?;
        blurred.push(b0);

        for i in 1..n_blur {
            let sigma = config.base_sigma * config.scale_factor.powi(i as i32);
            let radius = (sigma * 3.0).ceil() as u32;
            let level = Texture::intermediate_r32float(ctx.device(), w, h)?;
            blur.apply(ctx, input, &level, &GaussianConfig { sigma, radius })?;
            blurred.push(level);
        }

        let n_dog = n_blur - 1;
        let mut dogs: Vec<Texture> = Vec::with_capacity(n_dog);
        for i in 0..n_dog {
            let dog = Texture::intermediate_r32float(ctx.device(), w, h)?;
            self.subtract(ctx, &blurred[i], &blurred[i + 1], &dog, w, h)?;
            dogs.push(dog);
        }

        let kp_buf =
            vx_gpu::UnifiedBuffer::<DoGKeypoint>::new(ctx.device(), config.max_keypoints as usize)?;
        let mut count_buf = vx_gpu::UnifiedBuffer::<u32>::new(ctx.device(), 1)?;
        count_buf.as_mut_slice()[0] = 0;

        for i in 0..(n_dog.saturating_sub(2)) {
            let extrema_params = DoGExtremaParams {
                width: w,
                height: h,
                contrast_threshold: config.contrast_threshold,
                max_keypoints: config.max_keypoints,
                octave: 0,
                level: i as u32,
            };

            let cmd_buf = ctx
                .queue()
                .commandBuffer()
                .ok_or(Error::Gpu("failed to create command buffer".into()))?;
            let encoder = cmd_buf
                .computeCommandEncoder()
                .ok_or(Error::Gpu("failed to create compute encoder".into()))?;

            unsafe {
                encoder.setComputePipelineState(&self.extrema_pipeline);
                encoder.setTexture_atIndex(Some(dogs[i].raw()), 0);
                encoder.setTexture_atIndex(Some(dogs[i + 1].raw()), 1);
                encoder.setTexture_atIndex(Some(dogs[i + 2].raw()), 2);
                encoder.setBuffer_offset_atIndex(Some(kp_buf.metal_buffer()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(count_buf.metal_buffer()), 0, 1);
                encoder.setBytes_length_atIndex(
                    NonNull::new_unchecked(
                        &extrema_params as *const DoGExtremaParams as *mut c_void,
                    ),
                    mem::size_of::<DoGExtremaParams>(),
                    2,
                );

                let tew = self.extrema_pipeline.threadExecutionWidth();
                let max_tg = self.extrema_pipeline.maxTotalThreadsPerThreadgroup();
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
        }

        let n_found = count_buf.as_slice()[0] as usize;
        let n_found = n_found.min(config.max_keypoints as usize);
        Ok(kp_buf.as_slice()[..n_found].to_vec())
    }

    /// Encodes a single DoG subtraction without committing.
    pub fn encode_subtract(
        &self,
        cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
        blur_a: &Texture,
        blur_b: &Texture,
        output: &Texture,
        w: u32,
        h: u32,
    ) -> Result<()> {
        let params = DoGParams {
            width: w,
            height: h,
        };

        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or(Error::Gpu("failed to create compute encoder".into()))?;

        unsafe {
            encoder.setComputePipelineState(&self.subtract_pipeline);
            encoder.setTexture_atIndex(Some(blur_a.raw()), 0);
            encoder.setTexture_atIndex(Some(blur_b.raw()), 1);
            encoder.setTexture_atIndex(Some(output.raw()), 2);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(&params as *const DoGParams as *mut c_void),
                mem::size_of::<DoGParams>(),
                0,
            );

            let tew = self.subtract_pipeline.threadExecutionWidth();
            let max_tg = self.subtract_pipeline.maxTotalThreadsPerThreadgroup();
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
        Ok(())
    }

    fn subtract(
        &self,
        ctx: &Context,
        blur_a: &Texture,
        blur_b: &Texture,
        output: &Texture,
        w: u32,
        h: u32,
    ) -> Result<()> {
        let params = DoGParams {
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
            encoder.setComputePipelineState(&self.subtract_pipeline);
            encoder.setTexture_atIndex(Some(blur_a.raw()), 0);
            encoder.setTexture_atIndex(Some(blur_b.raw()), 1);
            encoder.setTexture_atIndex(Some(output.raw()), 2);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(&params as *const DoGParams as *mut c_void),
                mem::size_of::<DoGParams>(),
                0,
            );

            let tew = self.subtract_pipeline.threadExecutionWidth();
            let max_tg = self.subtract_pipeline.maxTotalThreadsPerThreadgroup();
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
        Ok(())
    }
}

unsafe impl Send for DoGDetector {}
unsafe impl Sync for DoGDetector {}
