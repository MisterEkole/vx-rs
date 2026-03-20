//! Euclidean distance transform via Jump Flooding Algorithm.

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
use crate::types::JFAParams;

/// Configuration for the distance transform.
#[derive(Clone, Debug)]
pub struct DistanceConfig {
    /// Intensity threshold (0.0--1.0); pixels at or below are seeds.
    pub threshold: f32,
}

impl Default for DistanceConfig {
    fn default() -> Self {
        Self { threshold: 0.5 }
    }
}

/// Distance transform compute pipelines.
pub struct DistanceTransform {
    init_pipeline:     Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    step_pipeline:     Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    distance_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl DistanceTransform {
    pub fn new(ctx: &Context) -> Result<Self, String> {
        let init_name = objc2_foundation::ns_string!("jfa_init");
        let step_name = objc2_foundation::ns_string!("jfa_step");
        let dist_name = objc2_foundation::ns_string!("jfa_distance");

        let init_func = ctx.library().newFunctionWithName(init_name)
            .ok_or_else(|| "Missing kernel function 'jfa_init'".to_string())?;
        let step_func = ctx.library().newFunctionWithName(step_name)
            .ok_or_else(|| "Missing kernel function 'jfa_step'".to_string())?;
        let dist_func = ctx.library().newFunctionWithName(dist_name)
            .ok_or_else(|| "Missing kernel function 'jfa_distance'".to_string())?;

        let init_pipeline = ctx.device()
            .newComputePipelineStateWithFunction_error(&init_func)
            .map_err(|e| format!("Failed to create jfa_init pipeline: {e}"))?;
        let step_pipeline = ctx.device()
            .newComputePipelineStateWithFunction_error(&step_func)
            .map_err(|e| format!("Failed to create jfa_step pipeline: {e}"))?;
        let distance_pipeline = ctx.device()
            .newComputePipelineStateWithFunction_error(&dist_func)
            .map_err(|e| format!("Failed to create jfa_distance pipeline: {e}"))?;

        Ok(Self { init_pipeline, step_pipeline, distance_pipeline })
    }

    /// Computes the distance transform, returning an R32Float texture.
    pub fn compute(
        &self,
        ctx:    &Context,
        input:  &Texture,
        config: &DistanceConfig,
    ) -> Result<Texture, String> {
        let w = input.width();
        let h = input.height();
        let n_pixels = (w as usize) * (h as usize);

        // Ping-pong seed buffers (uint2 per pixel)
        let seeds_a = vx_core::UnifiedBuffer::<[u32; 2]>::new(ctx.device(), n_pixels)?;
        let seeds_b = vx_core::UnifiedBuffer::<[u32; 2]>::new(ctx.device(), n_pixels)?;

        // Initialize seeds
        {
            let params = JFAParams {
                width: w, height: h, step_size: 0, threshold: config.threshold,
            };

            let cmd_buf = ctx.queue().commandBuffer()
                .ok_or("Failed to create command buffer")?;
            let encoder = cmd_buf.computeCommandEncoder()
                .ok_or("Failed to create compute encoder")?;

            unsafe {
                encoder.setComputePipelineState(&self.init_pipeline);
                encoder.setTexture_atIndex(Some(input.raw()), 0);
                encoder.setBuffer_offset_atIndex(Some(seeds_a.metal_buffer()), 0, 0);
                encoder.setBytes_length_atIndex(
                    NonNull::new_unchecked(&params as *const JFAParams as *mut c_void),
                    mem::size_of::<JFAParams>(),
                    1,
                );

                let tew    = self.init_pipeline.threadExecutionWidth();
                let max_tg = self.init_pipeline.maxTotalThreadsPerThreadgroup();
                let tg_h   = (max_tg / tew).max(1);
                let grid    = MTLSize { width: w as usize, height: h as usize, depth: 1 };
                let tg_size = MTLSize { width: tew,        height: tg_h,       depth: 1 };
                encoder.dispatchThreads_threadsPerThreadgroup(grid, tg_size);
            }

            encoder.endEncoding();
            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();
        }

        // JFA iterations: k = N/2, N/4, ..., 1
        let max_dim = w.max(h);
        let mut k = max_dim.next_power_of_two() as i32 / 2;
        // Ping-pong between seed buffers
        let mut read_a = true;

        while k >= 1 {
            let params = JFAParams {
                width: w, height: h, step_size: k, threshold: config.threshold,
            };

            let (src_buf, dst_buf) = if read_a {
                (seeds_a.metal_buffer(), seeds_b.metal_buffer())
            } else {
                (seeds_b.metal_buffer(), seeds_a.metal_buffer())
            };

            let cmd_buf = ctx.queue().commandBuffer()
                .ok_or("Failed to create command buffer")?;
            let encoder = cmd_buf.computeCommandEncoder()
                .ok_or("Failed to create compute encoder")?;

            unsafe {
                encoder.setComputePipelineState(&self.step_pipeline);
                encoder.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(dst_buf), 0, 1);
                encoder.setBytes_length_atIndex(
                    NonNull::new_unchecked(&params as *const JFAParams as *mut c_void),
                    mem::size_of::<JFAParams>(),
                    2,
                );

                let tew    = self.step_pipeline.threadExecutionWidth();
                let max_tg = self.step_pipeline.maxTotalThreadsPerThreadgroup();
                let tg_h   = (max_tg / tew).max(1);
                let grid    = MTLSize { width: w as usize, height: h as usize, depth: 1 };
                let tg_size = MTLSize { width: tew,        height: tg_h,       depth: 1 };
                encoder.dispatchThreads_threadsPerThreadgroup(grid, tg_size);
            }

            encoder.endEncoding();
            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();

            read_a = !read_a;
            k /= 2;
        }

        // Convert seed map to Euclidean distances
        let output = Texture::output_r32float(ctx.device(), w, h)?;

        let final_seeds = if read_a {
            seeds_a.metal_buffer()
        } else {
            seeds_b.metal_buffer()
        };

        let params = JFAParams {
            width: w, height: h, step_size: 0, threshold: config.threshold,
        };

        let cmd_buf = ctx.queue().commandBuffer()
            .ok_or("Failed to create command buffer")?;
        let encoder = cmd_buf.computeCommandEncoder()
            .ok_or("Failed to create compute encoder")?;

        unsafe {
            encoder.setComputePipelineState(&self.distance_pipeline);
            encoder.setBuffer_offset_atIndex(Some(final_seeds), 0, 0);
            encoder.setTexture_atIndex(Some(output.raw()), 0);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(&params as *const JFAParams as *mut c_void),
                mem::size_of::<JFAParams>(),
                1,
            );

            let tew    = self.distance_pipeline.threadExecutionWidth();
            let max_tg = self.distance_pipeline.maxTotalThreadsPerThreadgroup();
            let tg_h   = (max_tg / tew).max(1);
            let grid    = MTLSize { width: w as usize, height: h as usize, depth: 1 };
            let tg_size = MTLSize { width: tew,        height: tg_h,       depth: 1 };
            encoder.dispatchThreads_threadsPerThreadgroup(grid, tg_size);
        }

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        Ok(output)
    }
}
