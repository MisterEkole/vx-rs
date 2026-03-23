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

use vx_gpu::UnifiedBuffer;
use crate::context::Context;
use crate::error::{Error, Result};
use crate::texture::Texture;
use crate::types::{CornerPoint, HarrisParams};

/// Configuration for the Harris corner response scorer.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct HarrisConfig {
    /// Sensitivity parameter. Typical: 0.04--0.06.
    pub k: f32,

    /// Half-size of the structure tensor integration window (e.g. 3 = 7x7 patch).
    pub patch_radius: i32,
}

impl HarrisConfig {
    /// Creates a config with the given sensitivity and patch radius.
    pub fn new(k: f32, patch_radius: i32) -> Self {
        Self { k, patch_radius }
    }
}

impl Default for HarrisConfig {
    fn default() -> Self {
        Self {
            k: 0.04,
            patch_radius: 3,
        }
    }
}

/// Compiled Harris response pipeline.
pub struct HarrisScorer {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

unsafe impl Send for HarrisScorer {}
unsafe impl Sync for HarrisScorer {}

impl HarrisScorer {
    /// Compiles the Harris response pipeline.
    pub fn new(ctx: &Context) -> Result<Self> {
        let name = objc2_foundation::ns_string!("harris_response");

        let func = ctx.library().newFunctionWithName(name)
            .ok_or(Error::ShaderMissing("harris_response".into()))?;

        let pipeline = ctx.device().newComputePipelineStateWithFunction_error(&func)
            .map_err(|e| Error::PipelineCompile(format!("harris_response: {e}")))?;

        Ok(Self { pipeline })
    }

    /// Scores corners with Harris response R = det(M) - k*trace(M)^2. Synchronous.
    pub fn compute(
        &self,
        ctx: &Context,
        texture: &Texture,
        corners: &[CornerPoint],
        config: &HarrisConfig,
    ) -> Result<Vec<CornerPoint>> {
        if corners.is_empty() {
            return Ok(Vec::new());
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

        let _corner_guard = corner_buf.gpu_guard();

        let cmd_buf = ctx.queue().commandBuffer()
            .ok_or(Error::Gpu("failed to create command buffer".into()))?;

        let encoder = cmd_buf.computeCommandEncoder()
            .ok_or(Error::Gpu("failed to create compute encoder".into()))?;

        Self::encode_into(&self.pipeline, &encoder, texture, &corner_buf, &params, n_corners);

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        drop(_corner_guard);

        Ok(corner_buf.as_slice()[..n_corners].to_vec())
    }

    /// Encodes Harris scoring without committing.
    pub fn encode(
        &self,
        ctx: &Context,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        texture: &Texture,
        corners: &[CornerPoint],
        config: &HarrisConfig,
    ) -> Result<HarrisEncodedBuffers> {
        if corners.is_empty() {
            return Err(Error::InvalidConfig("cannot encode Harris with zero corners".into()));
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

    /// Reads scored corners after the command buffer has completed.
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

/// Buffers returned by [`HarrisScorer::encode`]. Must outlive the command buffer.
pub struct HarrisEncodedBuffers {
    pub corners: UnifiedBuffer<CornerPoint>,
    pub n_corners: usize,
}
