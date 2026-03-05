// vx-vision/src/kernels/stereomatch.rs
//
// Rust binding for the stereo feature matcher (StereoMatch.metal).
//
// Two-pass GPU pipeline:
//   Pass 1 (stereo_hamming)  – builds an n_left × n_right Hamming distance
//                              matrix, applying epipolar and disparity
//                              constraints. Invalid pairs are marked 0xFFFF.
//   Pass 2 (stereo_extract)  – scans each row for the best match, applies a
//                              Hamming threshold, triangulates depth, and
//                              atomically appends valid StereoMatchResult entries.
//
// Usage:
//   let ctx = Context::new()?;
//   let matcher = StereoMatcher::new(&ctx)?;
//   let result = matcher.run(
//       &ctx, &left_kpts, &right_kpts,
//       &left_descs, &right_descs,
//       &StereoConfig::default(),
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
use crate::types::{CornerPoint, ORBOutput, StereoMatchResult, StereoParams};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// User-facing config for the stereo feature matcher.
#[derive(Clone, Debug)]
pub struct StereoConfig {
    /// Maximum y-difference (pixels) for the epipolar constraint.
    /// Use 2.0 for well-rectified stereo pairs.
    pub max_epipolar: f32,

    /// Minimum valid disparity (pixels).
    pub min_disparity: f32,

    /// Maximum valid disparity (pixels).
    pub max_disparity: f32,

    /// Maximum Hamming distance accepted as a valid match (0–256).
    pub max_hamming: u32,

    /// Lowe's ratio test threshold (0.0–1.0). Lower = stricter.
    pub ratio_thresh: f32,

    /// Focal length x (pixels).
    pub fx: f32,

    /// Focal length y (pixels).
    pub fy: f32,

    /// Principal point x (pixels).
    pub cx: f32,

    /// Principal point y (pixels).
    pub cy: f32,

    /// Stereo baseline (metres).
    pub baseline: f32,
}

impl Default for StereoConfig {
    fn default() -> Self {
        Self {
            max_epipolar:  2.0,
            min_disparity: 1.0,
            max_disparity: 120.0,
            max_hamming:   50,
            ratio_thresh:  0.8,
            fx:            600.0,
            fy:            600.0,
            cx:            320.0,
            cy:            240.0,
            baseline:      0.12,
        }
    }
}

// ---------------------------------------------------------------------------
// Matcher
// ---------------------------------------------------------------------------

/// Compiled stereo matching pipeline. Create once, reuse across frames.
pub struct StereoMatcher {
    hamming_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    extract_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

/// Result of a stereo matching pass.
#[derive(Debug)]
pub struct StereoResult {
    /// Valid stereo matches with disparity and triangulated 3D position.
    pub matches: Vec<StereoMatchResult>,
}

impl StereoMatcher {
    // --------------------------------------------------------------------
    // Construction
    // --------------------------------------------------------------------

    /// Build both compute pipelines from the context's shader library.
    pub fn new(ctx: &Context) -> Result<Self, String> {
        let hamming_name = objc2_foundation::ns_string!("stereo_hamming");
        let extract_name = objc2_foundation::ns_string!("stereo_extract");

        let hamming_func = ctx.library().newFunctionWithName(hamming_name)
            .ok_or_else(|| "Missing kernel function 'stereo_hamming'".to_string())?;
        let extract_func = ctx.library().newFunctionWithName(extract_name)
            .ok_or_else(|| "Missing kernel function 'stereo_extract'".to_string())?;

        let hamming_pipeline = ctx.device()
            .newComputePipelineStateWithFunction_error(&hamming_func)
            .map_err(|e| format!("Failed to create stereo_hamming pipeline: {e}"))?;
        let extract_pipeline = ctx.device()
            .newComputePipelineStateWithFunction_error(&extract_func)
            .map_err(|e| format!("Failed to create stereo_extract pipeline: {e}"))?;

        Ok(Self { hamming_pipeline, extract_pipeline })
    }

    // --------------------------------------------------------------------
    // Synchronous dispatch
    // --------------------------------------------------------------------

    /// Match ORB features between left and right rectified images.
    ///
    /// `left_descs` / `right_descs` come from [`OrbDescriptor::compute`].
    /// Their 256-bit descriptor words are packed into GPU buffers internally;
    /// the `angle` field is not used by the matcher.
    ///
    /// Synchronous: encodes both passes in one command buffer, commits,
    /// waits for GPU completion, then reads back results.
    pub fn run(
        &self,
        ctx: &Context,
        left_kpts:  &[CornerPoint],
        right_kpts: &[CornerPoint],
        left_descs:  &[ORBOutput],
        right_descs: &[ORBOutput],
        config: &StereoConfig,
    ) -> Result<StereoResult, String> {
        assert_eq!(left_kpts.len(),  left_descs.len(),  "left_kpts and left_descs length mismatch");
        assert_eq!(right_kpts.len(), right_descs.len(), "right_kpts and right_descs length mismatch");

        let n_left  = left_kpts.len();
        let n_right = right_kpts.len();

        if n_left == 0 || n_right == 0 {
            return Ok(StereoResult { matches: Vec::new() });
        }

        // --- Pack raw descriptor words (8 × u32 per keypoint, no angle) ---
        // The shader indexes as desc[idx * 8 + word], so we strip the angle field.
        let left_words:  Vec<u32> = left_descs.iter().flat_map(|d| d.desc).collect();
        let right_words: Vec<u32> = right_descs.iter().flat_map(|d| d.desc).collect();

        let mut left_desc_buf: UnifiedBuffer<u32> =
            UnifiedBuffer::new(ctx.device(), left_words.len())?;
        left_desc_buf.write(&left_words);

        let mut right_desc_buf: UnifiedBuffer<u32> =
            UnifiedBuffer::new(ctx.device(), right_words.len())?;
        right_desc_buf.write(&right_words);

        // --- Keypoint buffers ---
        let mut left_kpt_buf: UnifiedBuffer<CornerPoint> =
            UnifiedBuffer::new(ctx.device(), n_left)?;
        left_kpt_buf.write(left_kpts);

        let mut right_kpt_buf: UnifiedBuffer<CornerPoint> =
            UnifiedBuffer::new(ctx.device(), n_right)?;
        right_kpt_buf.write(right_kpts);

        // --- Distance matrix (n_left × n_right × u16) ---
        let dist_matrix: UnifiedBuffer<u16> =
            UnifiedBuffer::new(ctx.device(), n_left * n_right)?;

        // --- Match output (worst-case: every left keypoint matches) ---
        let matches_buf: UnifiedBuffer<StereoMatchResult> =
            UnifiedBuffer::new(ctx.device(), n_left)?;

        let mut count_buf: UnifiedBuffer<u32> = UnifiedBuffer::new(ctx.device(), 1)?;
        count_buf.write(&[0u32]);

        let params = StereoParams {
            n_left:        n_left as u32,
            n_right:       n_right as u32,
            max_epipolar:  config.max_epipolar,
            min_disparity: config.min_disparity,
            max_disparity: config.max_disparity,
            max_hamming:   config.max_hamming,
            ratio_thresh:  config.ratio_thresh,
            fx:            config.fx,
            fy:            config.fy,
            cx:            config.cx,
            cy:            config.cy,
            baseline:      config.baseline,
        };

        // --- GPU guards ---
        let _ld = left_desc_buf.gpu_guard();
        let _rd = right_desc_buf.gpu_guard();
        let _lk = left_kpt_buf.gpu_guard();
        let _rk = right_kpt_buf.gpu_guard();
        let _dm = dist_matrix.gpu_guard();
        let _mt = matches_buf.gpu_guard();
        let _ct = count_buf.gpu_guard();

        // --- Encode both passes into a single command buffer ---
        let cmd_buf = ctx.queue().commandBuffer()
            .ok_or("Failed to create command buffer")?;

        // Pass 1: build Hamming distance matrix
        {
            let encoder = cmd_buf.computeCommandEncoder()
                .ok_or("Failed to create hamming encoder")?;

            Self::encode_hamming(
                &self.hamming_pipeline, &encoder,
                &left_desc_buf, &right_desc_buf,
                &left_kpt_buf, &right_kpt_buf,
                &dist_matrix, &params, n_left, n_right,
            );

            encoder.endEncoding();
        }

        // Pass 2: extract best match per left keypoint, triangulate
        {
            let encoder = cmd_buf.computeCommandEncoder()
                .ok_or("Failed to create extract encoder")?;

            Self::encode_extract(
                &self.extract_pipeline, &encoder,
                &dist_matrix, &left_kpt_buf, &right_kpt_buf,
                &matches_buf, &count_buf, &params, n_left,
            );

            encoder.endEncoding();
        }

        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        drop((_ld, _rd, _lk, _rk, _dm, _mt, _ct));

        // --- Readback ---
        let n_matches = (count_buf.as_slice()[0] as usize).min(n_left);
        let matches = matches_buf.as_slice()[..n_matches].to_vec();

        Ok(StereoResult { matches })
    }

    // --------------------------------------------------------------------
    // Internal: Pass 1 — Hamming distance matrix
    // --------------------------------------------------------------------

    fn encode_hamming(
        pipeline:    &ProtocolObject<dyn MTLComputePipelineState>,
        encoder:     &ProtocolObject<dyn MTLComputeCommandEncoder>,
        left_desc:   &UnifiedBuffer<u32>,
        right_desc:  &UnifiedBuffer<u32>,
        left_kpts:   &UnifiedBuffer<CornerPoint>,
        right_kpts:  &UnifiedBuffer<CornerPoint>,
        dist_matrix: &UnifiedBuffer<u16>,
        params:      &StereoParams,
        n_left:      usize,
        n_right:     usize,
    ) {
        // SAFETY: GPU encoder operations interact with device state.
        //
        // Binding matches StereoMatch.metal (stereo_hamming):
        //   buffer(0) = left_desc   (uint32_t*, n_left  × 8, read)
        //   buffer(1) = right_desc  (uint32_t*, n_right × 8, read)
        //   buffer(2) = left_kpts   (CornerPoint*, read)
        //   buffer(3) = right_kpts  (CornerPoint*, read)
        //   buffer(4) = dist_matrix (ushort*, n_left × n_right, write)
        //   buffer(5) = params      (StereoParams, constant)
        //
        // 2D dispatch: gid.x = left_idx, gid.y = right_idx.
        unsafe {
            encoder.setComputePipelineState(pipeline);
            encoder.setBuffer_offset_atIndex(Some(left_desc.metal_buffer()),   0, 0);
            encoder.setBuffer_offset_atIndex(Some(right_desc.metal_buffer()),  0, 1);
            encoder.setBuffer_offset_atIndex(Some(left_kpts.metal_buffer()),   0, 2);
            encoder.setBuffer_offset_atIndex(Some(right_kpts.metal_buffer()),  0, 3);
            encoder.setBuffer_offset_atIndex(Some(dist_matrix.metal_buffer()), 0, 4);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(params as *const StereoParams as *mut c_void),
                mem::size_of::<StereoParams>(),
                5,
            );

            let tew     = pipeline.threadExecutionWidth();
            let max_tg  = pipeline.maxTotalThreadsPerThreadgroup();
            let tg_h    = (max_tg / tew).max(1);

            let grid    = MTLSize { width: n_left,  height: n_right, depth: 1 };
            let tg_size = MTLSize { width: tew,     height: tg_h,   depth: 1 };

            encoder.dispatchThreads_threadsPerThreadgroup(grid, tg_size);
        }
    }

    // --------------------------------------------------------------------
    // Internal: Pass 2 — Extract best matches
    // --------------------------------------------------------------------

    fn encode_extract(
        pipeline:    &ProtocolObject<dyn MTLComputePipelineState>,
        encoder:     &ProtocolObject<dyn MTLComputeCommandEncoder>,
        dist_matrix: &UnifiedBuffer<u16>,
        left_kpts:   &UnifiedBuffer<CornerPoint>,
        right_kpts:  &UnifiedBuffer<CornerPoint>,
        matches:     &UnifiedBuffer<StereoMatchResult>,
        match_count: &UnifiedBuffer<u32>,
        params:      &StereoParams,
        n_left:      usize,
    ) {
        // SAFETY: GPU encoder operations interact with device state.
        //
        // Binding matches StereoMatch.metal (stereo_extract):
        //   buffer(0) = dist_matrix  (ushort*, read)
        //   buffer(1) = left_kpts    (CornerPoint*, read)
        //   buffer(2) = right_kpts   (CornerPoint*, read)
        //   buffer(3) = matches      (StereoMatchResult*, write)
        //   buffer(4) = match_count  (atomic_uint, read/write)
        //   buffer(5) = params       (StereoParams, constant)
        //
        // 1D dispatch: one thread per left keypoint.
        unsafe {
            encoder.setComputePipelineState(pipeline);
            encoder.setBuffer_offset_atIndex(Some(dist_matrix.metal_buffer()),  0, 0);
            encoder.setBuffer_offset_atIndex(Some(left_kpts.metal_buffer()),    0, 1);
            encoder.setBuffer_offset_atIndex(Some(right_kpts.metal_buffer()),   0, 2);
            encoder.setBuffer_offset_atIndex(Some(matches.metal_buffer()),      0, 3);
            encoder.setBuffer_offset_atIndex(Some(match_count.metal_buffer()),  0, 4);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(params as *const StereoParams as *mut c_void),
                mem::size_of::<StereoParams>(),
                5,
            );

            let tew     = pipeline.threadExecutionWidth();
            let grid    = MTLSize { width: n_left, height: 1, depth: 1 };
            let tg_size = MTLSize { width: tew,    height: 1, depth: 1 };

            encoder.dispatchThreads_threadsPerThreadgroup(grid, tg_size);
        }
    }
}
