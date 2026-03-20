//! Harris corner response scorer.

use core::ffi::c_void;
use core::ptr::NonNull;
use std::mem;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCommandQueue, MTLDevice, MTLLibrary,
    MTLSize,
};

use vx_core::UnifiedBuffer;
use crate::context::Context;
use crate::texture::Texture;
use crate::types::{CornerPoint, HarrisParams};

/// Configuration for the Harris corner response scorer.
#[derive(Clone, Debug)]
pub struct HarrisConfig {
    /// Sensitivity parameter. Typical: 0.04--0.06.
    pub k: f32,

    /// Half-size of the structure tensor integration window (e.g. 3 = 7x7 patch).
    pub patch_radius: i32,
}

impl Default for HarrisConfig {
    fn default() -> Self {
        Self {
            k: 0.04,
            patch_radius: 3,
        }
    }
}

/// Compiled Harris response pipeline. Reusable across frames.
pub struct HarrisScorer {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl HarrisScorer {
    /// Creates the compute pipeline from the context's shader library.

    pub fn new(ctx: &Context) -> Result<Self, String> {
        let name = objc2_foundation::ns_string!("harris_response");

        let func = ctx.library().newFunctionWithName(name)
            .ok_or_else(|| "Missing kernel function 'harris_response'".to_string())?;

        let pipeline = ctx.device().newComputePipelineStateWithFunction_error(&func)
            .map_err(|e| format!("Failed to create Harris pipeline: {e}"))?;

        Ok(Self { pipeline })
    }

    /// Scores candidate corners with the Harris response measure R = det(M) - k*trace(M)^2.
    ///
    /// Takes the source texture and a slice of corners (typically from FAST).
    /// Returns the same corners with `response` overwritten by the Harris score.
    ///
    /// Synchronous: encodes, commits, waits, then reads back.
    /// For pipelined usage, see [`Self::encode`].
    pub fn compute(
        &self,
        ctx: &Context,
        texture: &Texture,
        corners: &[CornerPoint],
        config: &HarrisConfig,
    ) -> Result<Vec<CornerPoint>, String> {
        if corners.is_empty() {
            return Ok(Vec::new());
        }

        let n_corners = corners.len();

        // The shader reads `.position` and writes `.response` in-place.
        let mut corner_buf: UnifiedBuffer<CornerPoint> =
            UnifiedBuffer::new(ctx.device(), n_corners)?;
        corner_buf.write(corners);

        let params = HarrisParams {
            n_corners: n_corners as u32,
            patch_radius: config.patch_radius,
            k: config.k,
        };

        let _corner_guard = corner_buf.gpu_guard();

        let cmd_buf = ctx.queue().commandBuffer()
            .ok_or("Failed to create command buffer")?;

        let encoder = cmd_buf.computeCommandEncoder()
            .ok_or("Failed to create compute encoder")?;

        Self::encode_into(&self.pipeline, &encoder, texture, &corner_buf, &params, n_corners);

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        drop(_corner_guard);

        Ok(corner_buf.as_slice()[..n_corners].to_vec())
    }

    /// Encodes Harris scoring into an existing compute encoder without committing.
    ///
    /// Typical chain: FAST encode -> Harris encode -> NMS encode -> commit.
    pub fn encode(
        &self,
        ctx: &Context,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        texture: &Texture,
        corners: &[CornerPoint],
        config: &HarrisConfig,
    ) -> Result<HarrisEncodedBuffers, String> {
        if corners.is_empty() {
            return Err("Cannot encode Harris with zero corners".into());
        }

        let n_corners = corners.len();

        let mut corner_buf: UnifiedBuffer<CornerPoint> =
            UnifiedBuffer::new(ctx.device(), n_corners)?;
        corner_buf.write(corners);

        let params = HarrisParams {
            n_corners: n_corners as u32,
            patch_radius: config.patch_radius,
            k: config.k,
        };

        Self::encode_into(&self.pipeline, encoder, texture, &corner_buf, &params, n_corners);

        Ok(HarrisEncodedBuffers {
            corners: corner_buf,
            n_corners,
        })
    }

    /// Reads scored corners from buffers returned by [`Self::encode`].
    ///
    /// Only valid after the command buffer has completed.
    pub fn read_results(buffers: &HarrisEncodedBuffers) -> Vec<CornerPoint> {
        buffers.corners.as_slice()[..buffers.n_corners].to_vec()
    }

    fn encode_into(
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        texture: &Texture,
        corner_buf: &UnifiedBuffer<CornerPoint>,
        params: &HarrisParams,
        n_corners: usize,
    ) {
        // SAFETY: setBytes requires a valid pointer; encoder ops interact with device state.
        //
        // Dispatch is 1D: one thread per corner (not per pixel like FAST).
        unsafe {
            encoder.setComputePipelineState(pipeline);
            encoder.setTexture_atIndex(Some(texture.raw()), 0);
            encoder.setBuffer_offset_atIndex(Some(corner_buf.metal_buffer()), 0, 0);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(params as *const HarrisParams as *mut c_void),
                mem::size_of::<HarrisParams>(),
                1,
            );

            let tew = pipeline.threadExecutionWidth();
            let grid = MTLSize {
                width: n_corners,
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

/// Buffers returned by [`HarrisScorer::encode`]. Keeps the corner buffer alive until readback.
pub struct HarrisEncodedBuffers {
    pub corners: UnifiedBuffer<CornerPoint>,
    pub n_corners: usize,
}
