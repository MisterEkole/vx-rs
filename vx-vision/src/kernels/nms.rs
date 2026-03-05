// vx-vision/src/kernels/nms.rs
//
// Rust binding for the Non-Maximum Suppression kernel (NMS.metal).
//
// NMS filters a set of corners so that no two surviving corners are
// within `min_distance` pixels of each other.  For each corner, if any
// other corner with a strictly higher response falls within that radius,
// the current corner is suppressed.  Compatible with both FAST and Harris
// corner outputs (both produce `CornerPoint` arrays).
//
// Typical pipeline:
//   FAST detect → Harris score → NMS → tracking / ORB describe
//
// Usage:
//   let ctx = Context::new()?;
//   let nms = NmsSuppressor::new(&ctx)?;
//   let survivors = nms.run(&ctx, &corners, &NmsConfig::default())?;

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
use crate::types::{CornerPoint, NMSParams};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// User-facing config for the NMS suppressor.
#[derive(Clone, Debug)]
pub struct NmsConfig {
    /// Minimum pixel distance between any two surviving corners.
    /// A corner is suppressed if a stronger corner exists within this radius.
    /// Typical values: 5–15 pixels.
    pub min_distance: f32,
}

impl Default for NmsConfig {
    fn default() -> Self {
        Self { min_distance: 8.0 }
    }
}

// ---------------------------------------------------------------------------
// Suppressor
// ---------------------------------------------------------------------------

/// Compiled NMS pipeline. Create once, reuse across frames.
pub struct NmsSuppressor {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl NmsSuppressor {
    // --------------------------------------------------------------------
    // Construction
    // --------------------------------------------------------------------

    /// Build the compute pipeline from the context's shader library.
    pub fn new(ctx: &Context) -> Result<Self, String> {
        let name = objc2_foundation::ns_string!("nms_suppress");

        let func = ctx.library().newFunctionWithName(name)
            .ok_or_else(|| "Missing kernel function 'nms_suppress'".to_string())?;

        let pipeline = ctx.device().newComputePipelineStateWithFunction_error(&func)
            .map_err(|e| format!("Failed to create NMS pipeline: {e}"))?;

        Ok(Self { pipeline })
    }

    // --------------------------------------------------------------------
    // Synchronous dispatch
    // --------------------------------------------------------------------

    /// Suppress non-maximal corners from `corners`.
    ///
    /// Returns surviving corners in non-deterministic order (GPU scheduling).
    /// Sort by `response` afterwards if you need a ranked list.
    ///
    /// Synchronous: encodes, commits, waits for GPU completion, reads back.
    /// For pipelined usage, use [`Self::encode`].
    pub fn run(
        &self,
        ctx: &Context,
        corners: &[CornerPoint],
        config:  &NmsConfig,
    ) -> Result<Vec<CornerPoint>, String> {
        if corners.is_empty() {
            return Ok(Vec::new());
        }

        let n = corners.len();

        let mut input_buf: UnifiedBuffer<CornerPoint> =
            UnifiedBuffer::new(ctx.device(), n)?;
        input_buf.write(corners);

        let output_buf: UnifiedBuffer<CornerPoint> =
            UnifiedBuffer::new(ctx.device(), n)?;

        let mut count_buf: UnifiedBuffer<u32> = UnifiedBuffer::new(ctx.device(), 1)?;
        count_buf.write(&[0u32]);

        let params = NMSParams {
            n_corners:    n as u32,
            min_distance: config.min_distance,
        };

        let _in_guard  = input_buf.gpu_guard();
        let _out_guard = output_buf.gpu_guard();
        let _ct_guard  = count_buf.gpu_guard();

        let cmd_buf = ctx.queue().commandBuffer()
            .ok_or("Failed to create command buffer")?;
        let encoder = cmd_buf.computeCommandEncoder()
            .ok_or("Failed to create compute encoder")?;

        Self::encode_into(&self.pipeline, &encoder, &input_buf, &output_buf, &count_buf, &params, n);

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        drop(_in_guard);
        drop(_out_guard);
        drop(_ct_guard);

        let n_out = (count_buf.as_slice()[0] as usize).min(n);
        Ok(output_buf.as_slice()[..n_out].to_vec())
    }

    // --------------------------------------------------------------------
    // Encode-only (for pipelining)
    // --------------------------------------------------------------------

    /// Encode NMS into an existing compute encoder without committing.
    /// Returns `NmsEncodedBuffers` holding all buffers alive until readback.
    pub fn encode(
        &self,
        ctx:     &Context,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        corners: &[CornerPoint],
        config:  &NmsConfig,
    ) -> Result<NmsEncodedBuffers, String> {
        if corners.is_empty() {
            return Err("Cannot encode NMS with zero corners".into());
        }

        let n = corners.len();

        let mut input_buf: UnifiedBuffer<CornerPoint> =
            UnifiedBuffer::new(ctx.device(), n)?;
        input_buf.write(corners);

        let output_buf: UnifiedBuffer<CornerPoint> =
            UnifiedBuffer::new(ctx.device(), n)?;

        let mut count_buf: UnifiedBuffer<u32> = UnifiedBuffer::new(ctx.device(), 1)?;
        count_buf.write(&[0u32]);

        let params = NMSParams {
            n_corners:    n as u32,
            min_distance: config.min_distance,
        };

        Self::encode_into(&self.pipeline, encoder, &input_buf, &output_buf, &count_buf, &params, n);

        Ok(NmsEncodedBuffers { input: input_buf, output: output_buf, count: count_buf, n_input: n })
    }

    /// Read surviving corners from buffers returned by [`Self::encode`].
    /// **Only call after the command buffer has completed.**
    pub fn read_results(buffers: &NmsEncodedBuffers) -> Vec<CornerPoint> {
        let n_out = (buffers.count.as_slice()[0] as usize).min(buffers.n_input);
        buffers.output.as_slice()[..n_out].to_vec()
    }

    // --------------------------------------------------------------------
    // Internal
    // --------------------------------------------------------------------

    fn encode_into(
        pipeline:   &ProtocolObject<dyn MTLComputePipelineState>,
        encoder:    &ProtocolObject<dyn MTLComputeCommandEncoder>,
        input_buf:  &UnifiedBuffer<CornerPoint>,
        output_buf: &UnifiedBuffer<CornerPoint>,
        count_buf:  &UnifiedBuffer<u32>,
        params:     &NMSParams,
        n:          usize,
    ) {
        // SAFETY: GPU encoder operations interact with device state.
        //
        // Binding matches NMS.metal (nms_suppress):
        //   buffer(0) = input     (CornerPoint[], read)
        //   buffer(1) = output    (CornerPoint[], write — atomic append)
        //   buffer(2) = out_count (atomic_uint)
        //   buffer(3) = params    (NMSParams, constant)
        //
        // Dispatch is 1D: one thread per input corner.
        unsafe {
            encoder.setComputePipelineState(pipeline);
            encoder.setBuffer_offset_atIndex(Some(input_buf.metal_buffer()),  0, 0);
            encoder.setBuffer_offset_atIndex(Some(output_buf.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(count_buf.metal_buffer()),  0, 2);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(params as *const NMSParams as *mut c_void),
                mem::size_of::<NMSParams>(),
                3,
            );

            let tew     = pipeline.threadExecutionWidth();
            let grid    = MTLSize { width: n,   height: 1, depth: 1 };
            let tg_size = MTLSize { width: tew, height: 1, depth: 1 };

            encoder.dispatchThreads_threadsPerThreadgroup(grid, tg_size);
        }
    }
}

/// Buffers returned by [`NmsSuppressor::encode`].
pub struct NmsEncodedBuffers {
    pub input:   UnifiedBuffer<CornerPoint>,
    pub output:  UnifiedBuffer<CornerPoint>,
    pub count:   UnifiedBuffer<u32>,
    pub n_input: usize,
}
