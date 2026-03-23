//! KLT (Kanade-Lucas-Tomasi) optical flow tracker.

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
use crate::types::KLTParams;
use vx_gpu::UnifiedBuffer;

/// Configuration for the KLT tracker.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct KltConfig {
    pub max_iterations: u32,
    pub epsilon: f32,
    pub win_radius: i32,
    pub max_level: u32,
    pub min_eigenvalue: f32,
}

impl Default for KltConfig {
    fn default() -> Self {
        Self {
            max_iterations: 15,
            epsilon: 0.01,
            win_radius: 5,
            max_level: 3,
            min_eigenvalue: 1e-4,
        }
    }
}

/// Four-level image pyramid (level 0 = full res, each subsequent level = half).
pub struct ImagePyramid {
    pub levels: [Texture; 4],
}

pub struct KltTracker {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

unsafe impl Send for KltTracker {}
unsafe impl Sync for KltTracker {}

/// Result of a single tracking pass.
#[derive(Debug)]
pub struct KltResult {
    /// Tracked positions in the current frame.
    pub points: Vec<[f32; 2]>,

    /// Per-point status: `true` = tracked, `false` = lost.
    pub status: Vec<bool>,
}

impl KltTracker {
    pub fn new(ctx: &Context) -> Result<Self> {
        let name = objc2_foundation::ns_string!("klt_track_forward");

        let func = ctx
            .library()
            .newFunctionWithName(name)
            .ok_or(Error::ShaderMissing("klt_track_forward".into()))?;

        let pipeline = ctx
            .device()
            .newComputePipelineStateWithFunction_error(&func)
            .map_err(|e| Error::PipelineCompile(format!("klt_track_forward: {e}")))?;

        Ok(Self { pipeline })
    }

    /// Tracks points between pyramid frames. Returns positions and status.
    pub fn track(
        &self,
        ctx: &Context,
        prev_pyramid: &ImagePyramid,
        curr_pyramid: &ImagePyramid,
        prev_points: &[[f32; 2]],
        config: &KltConfig,
    ) -> Result<KltResult> {
        if prev_points.is_empty() {
            return Ok(KltResult {
                points: Vec::new(),
                status: Vec::new(),
            });
        }

        let n_points = prev_points.len();

        let mut prev_buf: UnifiedBuffer<[f32; 2]> = UnifiedBuffer::new(ctx.device(), n_points)?;
        prev_buf.write(prev_points);

        let curr_buf: UnifiedBuffer<[f32; 2]> = UnifiedBuffer::new(ctx.device(), n_points)?;

        let status_buf: UnifiedBuffer<u8> = UnifiedBuffer::new(ctx.device(), n_points)?;

        let params = KLTParams {
            n_points: n_points as u32,
            max_iterations: config.max_iterations,
            epsilon: config.epsilon,
            win_radius: config.win_radius,
            max_level: config.max_level,
            min_eigenvalue: config.min_eigenvalue,
        };

        let _prev_guard = prev_buf.gpu_guard();
        let _curr_guard = curr_buf.gpu_guard();
        let _status_guard = status_buf.gpu_guard();

        let cmd_buf = ctx
            .queue()
            .commandBuffer()
            .ok_or(Error::Gpu("failed to create command buffer".into()))?;

        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or(Error::Gpu("failed to create compute encoder".into()))?;

        Self::encode_into(
            &self.pipeline,
            &encoder,
            prev_pyramid,
            curr_pyramid,
            &prev_buf,
            &curr_buf,
            &status_buf,
            &params,
            n_points,
        );

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        drop(_prev_guard);
        drop(_curr_guard);
        drop(_status_guard);

        let points = curr_buf.as_slice()[..n_points].to_vec();
        let status = status_buf.as_slice()[..n_points]
            .iter()
            .map(|&s| s != 0)
            .collect();

        Ok(KltResult { points, status })
    }

    /// Encodes KLT tracking without committing.
    pub fn encode(
        &self,
        ctx: &Context,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        prev_pyramid: &ImagePyramid,
        curr_pyramid: &ImagePyramid,
        prev_points: &[[f32; 2]],
        config: &KltConfig,
    ) -> Result<KltEncodedBuffers> {
        if prev_points.is_empty() {
            return Err(Error::InvalidConfig(
                "cannot encode KLT with zero points".into(),
            ));
        }

        let n_points = prev_points.len();

        let mut prev_buf: UnifiedBuffer<[f32; 2]> = UnifiedBuffer::new(ctx.device(), n_points)?;
        prev_buf.write(prev_points);

        let curr_buf: UnifiedBuffer<[f32; 2]> = UnifiedBuffer::new(ctx.device(), n_points)?;

        let status_buf: UnifiedBuffer<u8> = UnifiedBuffer::new(ctx.device(), n_points)?;

        let params = KLTParams {
            n_points: n_points as u32,
            max_iterations: config.max_iterations,
            epsilon: config.epsilon,
            win_radius: config.win_radius,
            max_level: config.max_level,
            min_eigenvalue: config.min_eigenvalue,
        };

        Self::encode_into(
            &self.pipeline,
            encoder,
            prev_pyramid,
            curr_pyramid,
            &prev_buf,
            &curr_buf,
            &status_buf,
            &params,
            n_points,
        );

        Ok(KltEncodedBuffers {
            prev_pts: prev_buf,
            curr_pts: curr_buf,
            status: status_buf,
            n_points,
        })
    }

    /// Reads tracked points after the command buffer has completed.
    pub fn read_results(buffers: &KltEncodedBuffers) -> KltResult {
        let points = buffers.curr_pts.as_slice()[..buffers.n_points].to_vec();
        let status = buffers.status.as_slice()[..buffers.n_points]
            .iter()
            .map(|&s| s != 0)
            .collect();
        KltResult { points, status }
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_into(
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        prev_pyramid: &ImagePyramid,
        curr_pyramid: &ImagePyramid,
        prev_buf: &UnifiedBuffer<[f32; 2]>,
        curr_buf: &UnifiedBuffer<[f32; 2]>,
        status_buf: &UnifiedBuffer<u8>,
        params: &KLTParams,
        n_points: usize,
    ) {
        unsafe {
            encoder.setComputePipelineState(pipeline);

            for (i, level) in prev_pyramid.levels.iter().enumerate() {
                encoder.setTexture_atIndex(Some(level.raw()), i);
            }
            for (i, level) in curr_pyramid.levels.iter().enumerate() {
                encoder.setTexture_atIndex(Some(level.raw()), 4 + i);
            }

            encoder.setBuffer_offset_atIndex(Some(prev_buf.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(curr_buf.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(status_buf.metal_buffer()), 0, 2);

            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(params as *const KLTParams as *mut c_void),
                mem::size_of::<KLTParams>(),
                3,
            );

            let tew = pipeline.threadExecutionWidth();
            let grid = MTLSize {
                width: n_points,
                height: 1,
                depth: 1,
            };
            let tg_size = MTLSize {
                width: tew,
                height: 1,
                depth: 1,
            };

            encoder.dispatchThreads_threadsPerThreadgroup(grid, tg_size);
        }
    }
}

/// Buffers returned by [`KltTracker::encode`]. Must outlive the command buffer.
pub struct KltEncodedBuffers {
    pub prev_pts: UnifiedBuffer<[f32; 2]>,
    pub curr_pts: UnifiedBuffer<[f32; 2]>,
    pub status: UnifiedBuffer<u8>,
    pub n_points: usize,
}
