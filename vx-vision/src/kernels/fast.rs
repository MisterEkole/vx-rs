// vx-vision/src/kernels/fast.rs
//
// Rust binding for the FAST-9 corner detector (FastDetect.metal).
//
// Usage:
//   let ctx = Context::new()?;
//   let fast = FastDetector::new(&ctx)?;
//   let result = fast.detect(&ctx, &texture, &FastDetectConfig::default())?;

use core::ffi::c_void;
use core::ptr::NonNull;
use std::mem;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLSize, MTLDevice, MTLLibrary, MTLCommandQueue,
};

use vx_core::UnifiedBuffer;
use crate::context::Context;
use crate::texture::Texture;
use crate::types::{CornerPoint, FASTParams};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// User-facing config for the FAST corner detector.
#[derive(Clone, Debug)]
pub struct FastDetectConfig {
    /// Intensity difference threshold for the circle test (0–255 range).
    /// Lower = more corners; higher = fewer, stronger corners.
    /// Typical: 10–40.
    pub threshold: i32,

    /// Maximum number of corners the output buffer can hold.
    pub max_corners: u32,
}

impl Default for FastDetectConfig {
    fn default() -> Self {
        Self {
            threshold: 20,
            max_corners: 8192,
        }
    }
}

// ---------------------------------------------------------------------------
// Detector
// ---------------------------------------------------------------------------

/// Compiled FAST-9 pipeline.  Create once, reuse across frames.
pub struct FastDetector {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

/// Result of a single detection pass.
#[derive(Debug)]
pub struct FastDetectResult {
    /// Detected corners.  Order is non-deterministic (GPU thread scheduling).
    /// Run NMS or sort by response on the CPU if you need ordering.
    pub corners: Vec<CornerPoint>,
}

impl FastDetector {
    // --------------------------------------------------------------------
    // Construction
    // --------------------------------------------------------------------

    /// Build the compute pipeline from the context's shader library.
    pub fn new(ctx: &Context) -> Result<Self, String> {
        let name = objc2_foundation::ns_string!("fast_detect");

        let func = ctx.library().newFunctionWithName(name)
            .ok_or_else(|| "Missing kernel function 'fast_detect'".to_string())?;

        let pipeline = ctx.device().newComputePipelineStateWithFunction_error(&func)
            .map_err(|e| format!("Failed to create FAST pipeline: {e}"))?;

        Ok(Self { pipeline })
    }

    // --------------------------------------------------------------------
    // Synchronous dispatch
    // --------------------------------------------------------------------

    /// Run the detector on a grayscale texture.
    ///
    /// Synchronous: encodes, commits, waits for GPU completion,
    /// then reads back results.  For pipelined usage, use [`Self::encode`].
    pub fn detect(
        &self,
        ctx: &Context,
        texture: &Texture,
        config: &FastDetectConfig,
    ) -> Result<FastDetectResult, String> {
        let width = texture.width();
        let height = texture.height();

        // --- Allocate shared-memory buffers via vx-core (zero-copy on UMA) ---
        let corner_buf: UnifiedBuffer<CornerPoint> =
            UnifiedBuffer::new(ctx.device(), config.max_corners as usize)?;
        let mut count_buf: UnifiedBuffer<u32> = UnifiedBuffer::new(ctx.device(), 1)?;

        // Zero the atomic counter
        count_buf.write(&[0u32]);

        // --- Pack params ---
        let params = FASTParams {
            threshold: config.threshold,
            max_corners: config.max_corners,
            width,
            height,
        };

        // --- Acquire GPU guards (prevent mutable CPU access during flight) ---
        let _corner_guard = corner_buf.gpu_guard();
        let _count_guard = count_buf.gpu_guard();

        // --- Encode & dispatch ---
        let cmd_buf = ctx.queue().commandBuffer()
            .ok_or("Failed to create command buffer")?;

        let encoder = cmd_buf.computeCommandEncoder()
            .ok_or("Failed to create compute encoder")?;

        Self::encode_into(
            &self.pipeline, &encoder, texture,
            &corner_buf, &count_buf, &params,
            width, height,
        );

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        // --- Guards drop here, releasing CPU access ---
        drop(_corner_guard);
        drop(_count_guard);

        // --- Readback  ---
        let n_detected = count_buf.as_slice()[0];
        let n = (n_detected as usize).min(config.max_corners as usize); //.min prevents reading garbage beyond the buffer

        let corners = corner_buf.as_slice()[..n].to_vec(); // slices first n valid corner into an owned Vec

        Ok(FastDetectResult { corners })
    }

    // --------------------------------------------------------------------
    // Encode-only (for pipelining)
    // --------------------------------------------------------------------

    /// Encode the FAST detection into an existing compute encoder
    /// without committing.  Returns `EncodedBuffers` holding the
    /// retained Metal buffers.
    ///
    /// Chain: FAST -> Harris -> NMS in a single command buffer.
    /// Read results with [`Self::read_results`] after commit + wait.
    pub fn encode(
        &self,
        ctx: &Context,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        texture: &Texture,
        config: &FastDetectConfig,
    ) -> Result<EncodedBuffers, String> {
        let width = texture.width();
        let height = texture.height();

        let corner_buf: UnifiedBuffer<CornerPoint> =
            UnifiedBuffer::new(ctx.device(), config.max_corners as usize)?;
        let mut count_buf: UnifiedBuffer<u32> = UnifiedBuffer::new(ctx.device(), 1)?;

        count_buf.write(&[0u32]);

        let params = FASTParams {
            threshold: config.threshold,
            max_corners: config.max_corners,
            width,
            height,
        };

        Self::encode_into(
            &self.pipeline, encoder, texture,
            &corner_buf, &count_buf, &params,
            width, height,
        );

        Ok(EncodedBuffers {
            corners: corner_buf,
            count: count_buf,
            max_corners: config.max_corners,
        })
    }

    /// Read corners from buffers returned by [`Self::encode`].
    /// **Only call after the command buffer has completed.**
    pub fn read_results(buffers: &EncodedBuffers) -> Vec<CornerPoint> {
        let n_detected = buffers.count.as_slice()[0];
        let n = (n_detected as usize).min(buffers.max_corners as usize);
        buffers.corners.as_slice()[..n].to_vec()
    }

    // --------------------------------------------------------------------
    // Internal
    // --------------------------------------------------------------------

    fn encode_into(
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        texture: &Texture,
        corner_buf: &UnifiedBuffer<CornerPoint>,
        count_buf: &UnifiedBuffer<u32>,
        params: &FASTParams,
        width: u32,
        height: u32,
    ) {
        // SAFETY: setBytes requires a valid NonNull pointer to param data.
        // GPU encoder operations interact with device state.
        unsafe {
            encoder.setComputePipelineState(pipeline);
            encoder.setTexture_atIndex(Some(texture.raw()), 0);
            encoder.setBuffer_offset_atIndex(Some(corner_buf.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(count_buf.metal_buffer()), 0, 1);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(params as *const FASTParams as *mut c_void),
                mem::size_of::<FASTParams>(),
                2,
            );

            let tew = pipeline.threadExecutionWidth();
            let max_tg = pipeline.maxTotalThreadsPerThreadgroup();

            let grid = MTLSize {
                width: width as usize,
                height: height as usize,
                depth: 1,
            };
            let tg_size = MTLSize {
                width: tew,
                height: (max_tg / tew).max(1),
                depth: 1,
            };

            encoder.dispatchThreads_threadsPerThreadgroup(grid, tg_size);
        }
    }
}

/// Buffers returned by [`FastDetector::encode`].
/// Holds the `UnifiedBuffer`s alive until readback.
pub struct EncodedBuffers {
    pub corners: UnifiedBuffer<CornerPoint>,
    pub count: UnifiedBuffer<u32>,
    pub max_corners: u32,
}