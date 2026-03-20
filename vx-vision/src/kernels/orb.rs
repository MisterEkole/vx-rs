//! ORB descriptor extractor (rotated BRIEF + intensity-centroid orientation).

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
use crate::types::{CornerPoint, ORBOutput, ORBParams};

/// Configuration for the ORB descriptor extractor.
#[derive(Clone, Debug)]
pub struct OrbConfig {
    /// Circular patch radius (default 15 for a 31x31 patch).
    pub patch_radius: u32,
}

impl Default for OrbConfig {
    fn default() -> Self {
        Self { patch_radius: 15 }
    }
}

/// Compiled ORB descriptor pipeline. Reusable across frames.
pub struct OrbDescriptor {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

/// Result of a single ORB extraction pass.
#[derive(Debug)]
pub struct OrbResult {
    /// Per-keypoint 256-bit descriptors and orientation angles.
    pub descriptors: Vec<ORBOutput>,
}

impl OrbDescriptor {
    /// Creates the compute pipeline from the context's shader library.
    pub fn new(ctx: &Context) -> Result<Self, String> {
        let name = objc2_foundation::ns_string!("orb_describe");

        let func = ctx.library().newFunctionWithName(name)
            .ok_or_else(|| "Missing kernel function 'orb_describe'".to_string())?;

        let pipeline = ctx.device().newComputePipelineStateWithFunction_error(&func)
            .map_err(|e| format!("Failed to create ORB pipeline: {e}"))?;

        Ok(Self { pipeline })
    }

    /// Computes ORB descriptors for a set of keypoints.
    ///
    /// `pattern` is 256 test pairs as `(dx1, dy1, dx2, dy2)`, flattened to 1024 `i32` values.
    ///
    /// Synchronous: encodes, commits, waits, then reads back.
    /// For pipelined usage, see [`Self::encode`].
    pub fn compute(
        &self,
        ctx: &Context,
        texture: &Texture,
        keypoints: &[CornerPoint],
        pattern: &[i32],
        config: &OrbConfig,
    ) -> Result<OrbResult, String> {
        if keypoints.is_empty() {
            return Ok(OrbResult { descriptors: Vec::new() });
        }
        assert_eq!(pattern.len(), 1024, "ORB pattern must be 1024 i32 values (256 pairs × 4)");

        let n_keypoints = keypoints.len();

        let mut kpt_buf: UnifiedBuffer<CornerPoint> =
            UnifiedBuffer::new(ctx.device(), n_keypoints)?;
        kpt_buf.write(keypoints);

        let out_buf: UnifiedBuffer<ORBOutput> =
            UnifiedBuffer::new(ctx.device(), n_keypoints)?;

        let mut pat_buf: UnifiedBuffer<i32> =
            UnifiedBuffer::new(ctx.device(), 1024)?;
        pat_buf.write(pattern);

        let params = ORBParams {
            n_keypoints: n_keypoints as u32,
            patch_radius: config.patch_radius,
        };

        let _kpt_guard = kpt_buf.gpu_guard();
        let _out_guard = out_buf.gpu_guard();
        let _pat_guard = pat_buf.gpu_guard();

        let cmd_buf = ctx.queue().commandBuffer()
            .ok_or("Failed to create command buffer")?;
        let encoder = cmd_buf.computeCommandEncoder()
            .ok_or("Failed to create compute encoder")?;

        Self::encode_into(
            &self.pipeline, &encoder, texture,
            &kpt_buf, &out_buf, &pat_buf, &params, n_keypoints,
        );

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        drop(_kpt_guard);
        drop(_out_guard);
        drop(_pat_guard);

        let descriptors = out_buf.as_slice()[..n_keypoints].to_vec();
        Ok(OrbResult { descriptors })
    }

    /// Encodes ORB extraction into an existing compute encoder without committing.
    ///
    /// Typical chain: FAST -> Harris -> ORB -> commit.
    pub fn encode(
        &self,
        ctx: &Context,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        texture: &Texture,
        keypoints: &[CornerPoint],
        pattern: &[i32],
        config: &OrbConfig,
    ) -> Result<OrbEncodedBuffers, String> {
        if keypoints.is_empty() {
            return Err("Cannot encode ORB with zero keypoints".into());
        }
        assert_eq!(pattern.len(), 1024, "ORB pattern must be 1024 i32 values");

        let n_keypoints = keypoints.len();

        let mut kpt_buf: UnifiedBuffer<CornerPoint> =
            UnifiedBuffer::new(ctx.device(), n_keypoints)?;
        kpt_buf.write(keypoints);

        let out_buf: UnifiedBuffer<ORBOutput> =
            UnifiedBuffer::new(ctx.device(), n_keypoints)?;

        let mut pat_buf: UnifiedBuffer<i32> =
            UnifiedBuffer::new(ctx.device(), 1024)?;
        pat_buf.write(pattern);

        let params = ORBParams {
            n_keypoints: n_keypoints as u32,
            patch_radius: config.patch_radius,
        };

        Self::encode_into(
            &self.pipeline, encoder, texture,
            &kpt_buf, &out_buf, &pat_buf, &params, n_keypoints,
        );

        Ok(OrbEncodedBuffers {
            keypoints: kpt_buf,
            descriptors: out_buf,
            pattern: pat_buf,
            n_keypoints,
        })
    }

    /// Reads ORB results from buffers returned by [`Self::encode`].
    ///
    /// Only valid after the command buffer has completed.
    pub fn read_results(buffers: &OrbEncodedBuffers) -> OrbResult {
        let descriptors = buffers.descriptors.as_slice()[..buffers.n_keypoints].to_vec();
        OrbResult { descriptors }
    }

    fn encode_into(
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        texture: &Texture,
        kpt_buf: &UnifiedBuffer<CornerPoint>,
        out_buf: &UnifiedBuffer<ORBOutput>,
        pat_buf: &UnifiedBuffer<i32>,
        params: &ORBParams,
        n_keypoints: usize,
    ) {
        // SAFETY: setBytes requires a valid pointer; encoder ops interact with device state.
        unsafe {
            encoder.setComputePipelineState(pipeline);
            encoder.setTexture_atIndex(Some(texture.raw()), 0);
            encoder.setBuffer_offset_atIndex(Some(kpt_buf.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(out_buf.metal_buffer()), 0, 1);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(params as *const ORBParams as *mut c_void),
                mem::size_of::<ORBParams>(),
                2,
            );
            encoder.setBuffer_offset_atIndex(Some(pat_buf.metal_buffer()), 0, 3);

            let tew = pipeline.threadExecutionWidth();
            let grid = MTLSize { width: n_keypoints, height: 1, depth: 1 };
            let tg_size = MTLSize { width: tew, height: 1, depth: 1 };

            encoder.dispatchThreads_threadsPerThreadgroup(grid, tg_size);
        }
    }
}

/// Buffers returned by [`OrbDescriptor::encode`]. Keeps allocations alive until readback.
pub struct OrbEncodedBuffers {
    pub keypoints:   UnifiedBuffer<CornerPoint>,
    pub descriptors: UnifiedBuffer<ORBOutput>,
    pub pattern:     UnifiedBuffer<i32>,
    pub n_keypoints: usize,
}
