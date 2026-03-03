// vx-vision/src/kernels/klt.rs
//
// Rust binding for the KLT (Kanade-Lucas-Tomasi) optical flow tracker
// (KLTTracker.metal).
//
// KLT tracks a set of points from a previous frame to a current frame
// using iterative Lucas-Kanade at multiple pyramid levels.  It requires
// pre-built image pyramids for both frames (4 levels each).
//
// Usage:
//   let ctx = Context::new()?;
//   let tracker = KltTracker::new(&ctx)?;
//   let result = tracker.track(
//       &ctx, &prev_pyramid, &curr_pyramid,
//       &points, &KltConfig::default(),
//   )?;

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

use vx_core::UnifiedBuffer;
use crate::context::Context;
use crate::texture::Texture;
use crate::types::KLTParams;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// User-facing config for the KLT optical flow tracker.
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

// ---------------------------------------------------------------------------
// Pyramid
// ---------------------------------------------------------------------------

/// A 4-level image pyramid for one frame.
///
/// Level 0 is the full-resolution image; each subsequent level is
/// half the dimensions of the previous.  Users must build pyramids
/// before calling the tracker.
///
/// The shader binds exactly 4 texture slots per frame, so we fix
/// the count at 4.  If `max_level < 3`, the unused levels are still
/// bound but never sampled by the shader.
pub struct ImagePyramid {
    pub levels: [Texture; 4],
}

// ---------------------------------------------------------------------------
// Tracker
// ---------------------------------------------------------------------------

pub struct KltTracker {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

/// Result of a single tracking pass.
#[derive(Debug)]
pub struct KltResult {
    /// Tracked point positions in the current frame.
    pub points: Vec<[f32; 2]>,

    /// Per-point tracking status.
    /// `true` = successfully tracked, `false` = lost (out of bounds or
    /// structure tensor below min_eigenvalue).
    pub status: Vec<bool>,
}

impl KltTracker {
    // --------------------------------------------------------------------
    // Construction-- compute pipeline
    // --------------------------------------------------------------------

    pub fn new(ctx: &Context) -> Result<Self, String> {
        let name = objc2_foundation::ns_string!("klt_track_forward");

        let func = ctx.library().newFunctionWithName(name)
            .ok_or_else(|| "Missing kernel function 'klt_track_forward'".to_string())?;

        let pipeline = ctx.device().newComputePipelineStateWithFunction_error(&func)
            .map_err(|e| format!("Failed to create KLT pipeline: {e}"))?;

        Ok(Self { pipeline })
    }

    // --------------------------------------------------------------------
    // Synchronous dispatch
    // --------------------------------------------------------------------

    /// Track points from the previous frame to the current frame.
    ///
    /// `prev_pyramid` and `curr_pyramid` are 4-level image pyramids.
    /// `prev_points` are the 2D positions to track (typically from FAST).
    ///
    /// Returns tracked positions and per-point success/failure status.
    pub fn track(
        &self,
        ctx: &Context,
        prev_pyramid: &ImagePyramid,
        curr_pyramid: &ImagePyramid,
        prev_points: &[[f32; 2]],
        config: &KltConfig,
    ) -> Result<KltResult, String> {
        if prev_points.is_empty() {
            return Ok(KltResult {
                points: Vec::new(),
                status: Vec::new(),
            });
        }

        let n_points = prev_points.len();

        // --- Allocate buffers ---
        // buffer(0): prev_pts  — input positions (read by shader)
        // buffer(1): curr_pts  — output positions (written by shader)
        // buffer(2): status    — per-point success flag (written by shader)
        let mut prev_buf: UnifiedBuffer<[f32; 2]> =
            UnifiedBuffer::new(ctx.device(), n_points)?;
        prev_buf.write(prev_points);

        let curr_buf: UnifiedBuffer<[f32; 2]> =
            UnifiedBuffer::new(ctx.device(), n_points)?;

        let status_buf: UnifiedBuffer<u8> =
            UnifiedBuffer::new(ctx.device(), n_points)?;

        // --- Pack params ---
        let params = KLTParams {
            n_points: n_points as u32,
            max_iterations: config.max_iterations,
            epsilon: config.epsilon,
            win_radius: config.win_radius,
            max_level: config.max_level,
            min_eigenvalue: config.min_eigenvalue,
        };

        // --- GPU guards on all three data buffers ---
        let _prev_guard = prev_buf.gpu_guard();
        let _curr_guard = curr_buf.gpu_guard();
        let _status_guard = status_buf.gpu_guard();

        // --- Encode & dispatch ---
        let cmd_buf = ctx.queue().commandBuffer()
            .ok_or("Failed to create command buffer")?;

        let encoder = cmd_buf.computeCommandEncoder()
            .ok_or("Failed to create compute encoder")?;

        Self::encode_into(
            &self.pipeline, &encoder,
            prev_pyramid, curr_pyramid,
            &prev_buf, &curr_buf, &status_buf,
            &params, n_points,
        );

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();


        drop(_prev_guard);
        drop(_curr_guard);
        drop(_status_guard);

        // --- Readback ---
        let points = curr_buf.as_slice()[..n_points].to_vec();
        let status = status_buf.as_slice()[..n_points]
            .iter()
            .map(|&s| s != 0)
            .collect();

        Ok(KltResult { points, status })
    }

    // --------------------------------------------------------------------
    // Encode-only (for pipelining)
    // --------------------------------------------------------------------

    /// Encode KLT tracking into an existing compute encoder without
    /// committing.  Returns `KltEncodedBuffers` holding all three
    /// buffers alive until readback.
    pub fn encode(
        &self,
        ctx: &Context,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        prev_pyramid: &ImagePyramid,
        curr_pyramid: &ImagePyramid,
        prev_points: &[[f32; 2]],
        config: &KltConfig,
    ) -> Result<KltEncodedBuffers, String> {
        if prev_points.is_empty() {
            return Err("Cannot encode KLT with zero points".into());
        }

        let n_points = prev_points.len();

        let mut prev_buf: UnifiedBuffer<[f32; 2]> =
            UnifiedBuffer::new(ctx.device(), n_points)?;
        prev_buf.write(prev_points);

        let curr_buf: UnifiedBuffer<[f32; 2]> =
            UnifiedBuffer::new(ctx.device(), n_points)?;

        let status_buf: UnifiedBuffer<u8> =
            UnifiedBuffer::new(ctx.device(), n_points)?;

        let params = KLTParams {
            n_points: n_points as u32,
            max_iterations: config.max_iterations,
            epsilon: config.epsilon,
            win_radius: config.win_radius,
            max_level: config.max_level,
            min_eigenvalue: config.min_eigenvalue,
        };

        Self::encode_into(
            &self.pipeline, encoder,
            prev_pyramid, curr_pyramid,
            &prev_buf, &curr_buf, &status_buf,
            &params, n_points,
        );

        Ok(KltEncodedBuffers {
            prev_pts: prev_buf,
            curr_pts: curr_buf,
            status: status_buf,
            n_points,
        })
    }

    /// Read tracked points from buffers returned by [`Self::encode`].
    /// **Only call after the command buffer has completed.**
    pub fn read_results(buffers: &KltEncodedBuffers) -> KltResult {
        let points = buffers.curr_pts.as_slice()[..buffers.n_points].to_vec();
        let status = buffers.status.as_slice()[..buffers.n_points]
            .iter()
            .map(|&s| s != 0)
            .collect();
        KltResult { points, status }
    }

    // --------------------------------------------------------------------
    // Internal
    // --------------------------------------------------------------------

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
        // SAFETY: GPU encoder operations interact with device state.
        //
        // Texture binding matches KLTTracker.metal:
        //   texture(0..3)  = prev pyramid levels 0–3
        //   texture(4..7)  = curr pyramid levels 0–3
        //
        // Buffer binding:
        //   buffer(0) = prev_pts   (float2, read)
        //   buffer(1) = curr_pts   (float2, write)
        //   buffer(2) = status     (uint8_t, write)
        //   buffer(3) = params
        //
        // Dispatch is 1D: one thread per point.
        unsafe {
            encoder.setComputePipelineState(pipeline);

            // Bind 8 pyramid textures
            
            for (i, level) in prev_pyramid.levels.iter().enumerate() {
                encoder.setTexture_atIndex(Some(level.raw()), i);
            }
            for (i, level) in curr_pyramid.levels.iter().enumerate() {
                encoder.setTexture_atIndex(Some(level.raw()), 4 + i);
            }

            // Bind data buffers
            encoder.setBuffer_offset_atIndex(Some(prev_buf.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(curr_buf.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(status_buf.metal_buffer()), 0, 2);

            // Bind params at buffer(3)
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(params as *const KLTParams as *mut c_void),
                mem::size_of::<KLTParams>(),
                3,
            );

            // 1D dispatch: one thread per point
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

/// Buffers returned by [`KltTracker::encode`].
/// Holds all three buffers alive until readback.
pub struct KltEncodedBuffers {
    pub prev_pts: UnifiedBuffer<[f32; 2]>,
    pub curr_pts: UnifiedBuffer<[f32; 2]>,
    pub status: UnifiedBuffer<u8>,
    pub n_points: usize,
}