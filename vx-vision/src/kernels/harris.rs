// vx-vision/src/kernels/harris.rs
//
// Rust binding for the Harris corner response scorer (HarrisResponse.metal).
//
// Harris refines corners found by FAST: it takes a list of candidate
// corners and replaces each corner's `response` field with the Harris
// corner measure R = det(M) - k·trace(M)².
//
// Usage:
//   let ctx = Context::new()?;
//   let harris = HarrisScorer::new(&ctx)?;
//   let scored = harris.compute(&ctx, &texture, &corners, &HarrisConfig::default())?;

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

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// User-facing config for the Harris corner response scorer.
#[derive(Clone, Debug)]
pub struct HarrisConfig {
    /// Harris sensitivity parameter.
    /// Lower = more sensitive to edges; typical value: 0.04–0.06.
    pub k: f32,

    /// Half-size of the integration window for the structure tensor.
    /// A value of 3 means a 7×7 patch (2·3+1 = 7).
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

// ---------------------------------------------------------------------------
// Scorer
// ---------------------------------------------------------------------------

/// Compiled Harris response pipeline. Create once, reuse across frames.
pub struct HarrisScorer {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl HarrisScorer {
    // --------------------------------------------------------------------
    // Construction
    // --------------------------------------------------------------------

    /// Build the compute pipeline from the context's shader library.
    
    pub fn new(ctx: &Context) -> Result<Self, String> {
        let name = objc2_foundation::ns_string!("harris_response");

        let func = ctx.library().newFunctionWithName(name)
            .ok_or_else(|| "Missing kernel function 'harris_response'".to_string())?;

        let pipeline = ctx.device().newComputePipelineStateWithFunction_error(&func)
            .map_err(|e| format!("Failed to create Harris pipeline: {e}"))?;

        Ok(Self { pipeline })
    }

    // --------------------------------------------------------------------
    // Synchronous dispatch
    // --------------------------------------------------------------------

    /// Score a set of candidate corners with the Harris response measure.
    ///
    /// Takes the original image texture (for gradient computation) and a
    /// slice of corners (typically from FAST). Returns the same corners
    /// with their `response` field overwritten by the Harris score.
    ///
    /// Synchronous: encodes, commits, waits for GPU completion, then
    /// reads back results. For pipelined usage, use [`Self::encode`].
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

        // --- Upload corners to shared-memory buffer  ---
        // The shader reads .position and writes .response in-place.
        let mut corner_buf: UnifiedBuffer<CornerPoint> =
            UnifiedBuffer::new(ctx.device(), n_corners)?;
        corner_buf.write(corners);

        // --- Pack params ---
        // Matches HarrisParams in HarrisResponse.metal:
        //   buffer(0) = corners (read/write)
        //   buffer(1) = params
        let params = HarrisParams {
            n_corners: n_corners as u32,
            patch_radius: config.patch_radius,
            k: config.k,
        };

        // --- GPU guard ---
        let _corner_guard = corner_buf.gpu_guard();

        // --- Encode & dispatch ---
        let cmd_buf = ctx.queue().commandBuffer()
            .ok_or("Failed to create command buffer")?;

        let encoder = cmd_buf.computeCommandEncoder()
            .ok_or("Failed to create compute encoder")?;

        Self::encode_into(&self.pipeline, &encoder, texture, &corner_buf, &params, n_corners);

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        // --- Release guard ---
        drop(_corner_guard);

        // --- Readback (same buffer, response fields now updated) ---
        Ok(corner_buf.as_slice()[..n_corners].to_vec())
    }

    // --------------------------------------------------------------------
    // Encode-only (for pipelining)
    // --------------------------------------------------------------------

    /// Encode Harris scoring into an existing compute encoder without
    /// committing. Returns `HarrisEncodedBuffers` holding the corner
    /// buffer alive until readback.
    ///
    /// Typical chain: FAST encode → Harris encode → NMS encode → commit.
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

    /// Read scored corners from buffers returned by [`Self::encode`].
    /// **Only call after the command buffer has completed.**
    pub fn read_results(buffers: &HarrisEncodedBuffers) -> Vec<CornerPoint> {
        buffers.corners.as_slice()[..buffers.n_corners].to_vec()
    }

    // --------------------------------------------------------------------
    // Internal
    // --------------------------------------------------------------------

    fn encode_into(
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        texture: &Texture,
        corner_buf: &UnifiedBuffer<CornerPoint>,
        params: &HarrisParams,
        n_corners: usize,
    ) {
        // SAFETY: setBytes requires a valid NonNull pointer to param data.
        // GPU encoder operations interact with device state.
        //
        // Buffer binding matches HarrisResponse.metal:
        //   texture(0) = image
        //   buffer(0)  = corners (read position, write response)
        //   buffer(1)  = params
        //
        // Dispatch is 1D: one thread per corner (not per pixel like FAST).
        unsafe {
            encoder.setComputePipelineState(pipeline);
            encoder.setTexture_atIndex(Some(texture.raw()), 0);
            encoder.setBuffer_offset_atIndex(Some(corner_buf.metal_buffer()), 0, 0);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(params as *const HarrisParams as *mut c_void),
                mem::size_of::<HarrisParams>(),
                1,  // buffer(1) — not buffer(2) like FAST!
            );

            // 1D dispatch: one thread per corner
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

/// Buffers returned by [`HarrisScorer::encode`].
/// Holds the corner buffer alive until readback.
pub struct HarrisEncodedBuffers {
    pub corners: UnifiedBuffer<CornerPoint>,
    pub n_corners: usize,
}