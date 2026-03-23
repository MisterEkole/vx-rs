//! FAST-9 corner detector.

use core::ffi::c_void;
use core::ptr::NonNull;
use std::mem;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLSize, MTLDevice, MTLLibrary, MTLCommandQueue,
};

use vx_gpu::UnifiedBuffer;
use crate::context::Context;
use crate::error::{Error, Result};
use crate::texture::Texture;
use crate::types::{CornerPoint, FASTParams};

/// Configuration for the FAST corner detector.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct FastDetectConfig {
    /// Intensity difference threshold (0--255). Lower values yield more corners.
    pub threshold: i32,

    /// Maximum number of corners the output buffer can hold.
    pub max_corners: u32,
}

impl FastDetectConfig {
    /// Creates a config with the given threshold and max corners.
    pub fn new(threshold: i32, max_corners: u32) -> Self {
        Self { threshold, max_corners }
    }
}

impl Default for FastDetectConfig {
    fn default() -> Self {
        Self {
            threshold: 20,
            max_corners: 8192,
        }
    }
}

/// Compiled FAST-9 pipeline.
pub struct FastDetector {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

unsafe impl Send for FastDetector {}
unsafe impl Sync for FastDetector {}

/// Result of a single detection pass.
#[derive(Debug)]
pub struct FastDetectResult {
    /// Detected corners in non-deterministic order (GPU scheduling).
    pub corners: Vec<CornerPoint>,
}

impl FastDetector {
    /// Compiles the FAST-9 compute pipeline.
    pub fn new(ctx: &Context) -> Result<Self> {
        let name = objc2_foundation::ns_string!("fast_detect");

        let func = ctx.library().newFunctionWithName(name)
            .ok_or(Error::ShaderMissing("fast_detect".into()))?;

        let pipeline = ctx.device().newComputePipelineStateWithFunction_error(&func)
            .map_err(|e| Error::PipelineCompile(format!("fast_detect: {e}")))?;

        Ok(Self { pipeline })
    }

    /// Detects corners in a grayscale texture. Synchronous.
    pub fn detect(
        &self,
        ctx: &Context,
        texture: &Texture,
        config: &FastDetectConfig,
    ) -> Result<FastDetectResult> {
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

        let _corner_guard = corner_buf.gpu_guard();
        let _count_guard = count_buf.gpu_guard();

        let cmd_buf = ctx.queue().commandBuffer()
            .ok_or(Error::Gpu("failed to create command buffer".into()))?;

        let encoder = cmd_buf.computeCommandEncoder()
            .ok_or(Error::Gpu("failed to create compute encoder".into()))?;

        Self::encode_into(
            &self.pipeline, &encoder, texture,
            &corner_buf, &count_buf, &params,
            width, height,
        );

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        drop(_corner_guard);
        drop(_count_guard);

        let n_detected = count_buf.as_slice()[0];
        let n = (n_detected as usize).min(config.max_corners as usize);

        let corners = corner_buf.as_slice()[..n].to_vec();

        Ok(FastDetectResult { corners })
    }

    /// Encodes FAST detection without committing. Read with [`Self::read_results`] after commit.
    pub fn encode(
        &self,
        ctx: &Context,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        texture: &Texture,
        config: &FastDetectConfig,
    ) -> Result<EncodedBuffers> {
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

    /// Reads corners after the command buffer has completed.
    pub fn read_results(buffers: &EncodedBuffers) -> Vec<CornerPoint> {
        let n_detected = buffers.count.as_slice()[0];
        let n = (n_detected as usize).min(buffers.max_corners as usize);
        buffers.corners.as_slice()[..n].to_vec()
    }

    #[allow(clippy::too_many_arguments)]
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

/// Buffers returned by [`FastDetector::encode`]. Must outlive the command buffer.
pub struct EncodedBuffers {
    pub corners: UnifiedBuffer<CornerPoint>,
    pub count: UnifiedBuffer<u32>,
    pub max_corners: u32,
}
