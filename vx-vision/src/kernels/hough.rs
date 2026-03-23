//! Hough line transform.
//!
//! No `encode()` method is provided because Hough requires a CPU readback
//! of the accumulator buffer to extract peak lines after the voting pass.

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
use crate::types::{HoughLine, HoughPeakParams, HoughVoteParams};

/// Configuration for Hough line detection.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct HoughConfig {
    /// Angle bins over [0, pi). Typically 180.
    pub n_theta: u32,
    /// Minimum pixel intensity to count as an edge (0.0--1.0).
    pub edge_threshold: f32,
    /// Minimum vote count for a line.
    pub vote_threshold: u32,
    /// Maximum number of lines detected.
    pub max_lines: u32,
    /// Non-maximum suppression radius in accumulator space.
    pub nms_radius: u32,
}

impl Default for HoughConfig {
    fn default() -> Self {
        Self {
            n_theta: 180,
            edge_threshold: 0.5,
            vote_threshold: 100,
            max_lines: 128,
            nms_radius: 5,
        }
    }
}

/// Hough line detection pipelines. Requires CPU readback between passes.
pub struct HoughLines {
    vote_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    peak_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl HoughLines {
    pub fn new(ctx: &Context) -> Result<Self> {
        let vote_name = objc2_foundation::ns_string!("hough_vote");
        let peak_name = objc2_foundation::ns_string!("hough_peaks");

        let vote_func = ctx
            .library()
            .newFunctionWithName(vote_name)
            .ok_or(Error::ShaderMissing("hough_vote".into()))?;
        let peak_func = ctx
            .library()
            .newFunctionWithName(peak_name)
            .ok_or(Error::ShaderMissing("hough_peaks".into()))?;

        let vote_pipeline = ctx
            .device()
            .newComputePipelineStateWithFunction_error(&vote_func)
            .map_err(|e| Error::PipelineCompile(format!("hough_vote: {e}")))?;
        let peak_pipeline = ctx
            .device()
            .newComputePipelineStateWithFunction_error(&peak_func)
            .map_err(|e| Error::PipelineCompile(format!("hough_peaks: {e}")))?;

        Ok(Self {
            vote_pipeline,
            peak_pipeline,
        })
    }

    /// Detects lines in an edge image via the Hough transform.
    pub fn detect(
        &self,
        ctx: &Context,
        edge_image: &Texture,
        config: &HoughConfig,
    ) -> Result<Vec<HoughLine>> {
        let w = edge_image.width();
        let h = edge_image.height();
        let rho_max = ((w * w + h * h) as f32).sqrt();
        let n_rho = (2.0 * rho_max).ceil() as u32;

        let acc_size = (config.n_theta as usize) * (n_rho as usize);
        let mut acc_buf = vx_gpu::UnifiedBuffer::<u32>::new(ctx.device(), acc_size)?;
        for v in acc_buf.as_mut_slice().iter_mut() {
            *v = 0;
        }

        {
            let vote_params = HoughVoteParams {
                width: w,
                height: h,
                n_theta: config.n_theta,
                n_rho,
                rho_max,
                edge_threshold: config.edge_threshold,
            };

            let cmd_buf = ctx
                .queue()
                .commandBuffer()
                .ok_or(Error::Gpu("failed to create command buffer".into()))?;
            let encoder = cmd_buf
                .computeCommandEncoder()
                .ok_or(Error::Gpu("failed to create compute encoder".into()))?;

            unsafe {
                encoder.setComputePipelineState(&self.vote_pipeline);
                encoder.setTexture_atIndex(Some(edge_image.raw()), 0);
                encoder.setBuffer_offset_atIndex(Some(acc_buf.metal_buffer()), 0, 0);
                encoder.setBytes_length_atIndex(
                    NonNull::new_unchecked(&vote_params as *const HoughVoteParams as *mut c_void),
                    mem::size_of::<HoughVoteParams>(),
                    1,
                );

                let tew = self.vote_pipeline.threadExecutionWidth();
                let max_tg = self.vote_pipeline.maxTotalThreadsPerThreadgroup();
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

        let line_buf =
            vx_gpu::UnifiedBuffer::<HoughLine>::new(ctx.device(), config.max_lines as usize)?;
        let mut count_buf = vx_gpu::UnifiedBuffer::<u32>::new(ctx.device(), 1)?;
        count_buf.as_mut_slice()[0] = 0;

        {
            let peak_params = HoughPeakParams {
                n_theta: config.n_theta,
                n_rho,
                vote_threshold: config.vote_threshold,
                max_lines: config.max_lines,
                rho_max,
                nms_radius: config.nms_radius,
            };

            let cmd_buf = ctx
                .queue()
                .commandBuffer()
                .ok_or(Error::Gpu("failed to create command buffer".into()))?;
            let encoder = cmd_buf
                .computeCommandEncoder()
                .ok_or(Error::Gpu("failed to create compute encoder".into()))?;

            unsafe {
                encoder.setComputePipelineState(&self.peak_pipeline);
                encoder.setBuffer_offset_atIndex(Some(acc_buf.metal_buffer()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(line_buf.metal_buffer()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(count_buf.metal_buffer()), 0, 2);
                encoder.setBytes_length_atIndex(
                    NonNull::new_unchecked(&peak_params as *const HoughPeakParams as *mut c_void),
                    mem::size_of::<HoughPeakParams>(),
                    3,
                );

                let tew = self.peak_pipeline.threadExecutionWidth();
                let max_tg = self.peak_pipeline.maxTotalThreadsPerThreadgroup();
                let tg_h = (max_tg / tew).max(1);
                let grid = MTLSize {
                    width: config.n_theta as usize,
                    height: n_rho as usize,
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

        let n_lines = count_buf.as_slice()[0] as usize;
        let n_lines = n_lines.min(config.max_lines as usize);
        Ok(line_buf.as_slice()[..n_lines].to_vec())
    }
}

unsafe impl Send for HoughLines {}
unsafe impl Sync for HoughLines {}
