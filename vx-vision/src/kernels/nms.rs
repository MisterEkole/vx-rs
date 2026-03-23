//! Non-maximum suppression for corner points.

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
use crate::types::{CornerPoint, NMSParams};
use vx_gpu::UnifiedBuffer;

/// Configuration for the NMS suppressor.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct NmsConfig {
    /// Minimum pixel distance between surviving corners. Typical: 5--15.
    pub min_distance: f32,
}

impl NmsConfig {
    /// Creates a config with the given minimum distance.
    pub fn new(min_distance: f32) -> Self {
        Self { min_distance }
    }
}

impl Default for NmsConfig {
    fn default() -> Self {
        Self { min_distance: 8.0 }
    }
}

/// Compiled NMS pipeline.
pub struct NmsSuppressor {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

unsafe impl Send for NmsSuppressor {}
unsafe impl Sync for NmsSuppressor {}

impl NmsSuppressor {
    /// Compiles the NMS pipeline.
    pub fn new(ctx: &Context) -> Result<Self> {
        let name = objc2_foundation::ns_string!("nms_suppress");

        let func = ctx
            .library()
            .newFunctionWithName(name)
            .ok_or(Error::ShaderMissing("nms_suppress".into()))?;

        let pipeline = ctx
            .device()
            .newComputePipelineStateWithFunction_error(&func)
            .map_err(|e| Error::PipelineCompile(format!("nms_suppress: {e}")))?;

        Ok(Self { pipeline })
    }

    /// Suppresses non-maximal corners. Synchronous.
    pub fn run(
        &self,
        ctx: &Context,
        corners: &[CornerPoint],
        config: &NmsConfig,
    ) -> Result<Vec<CornerPoint>> {
        if corners.is_empty() {
            return Ok(Vec::new());
        }

        let n = corners.len();

        let mut input_buf: UnifiedBuffer<CornerPoint> = UnifiedBuffer::new(ctx.device(), n)?;
        input_buf.write(corners);

        let output_buf: UnifiedBuffer<CornerPoint> = UnifiedBuffer::new(ctx.device(), n)?;

        let mut count_buf: UnifiedBuffer<u32> = UnifiedBuffer::new(ctx.device(), 1)?;
        count_buf.write(&[0u32]);

        let params = NMSParams {
            n_corners: n as u32,
            min_distance: config.min_distance,
        };

        let _in_guard = input_buf.gpu_guard();
        let _out_guard = output_buf.gpu_guard();
        let _ct_guard = count_buf.gpu_guard();

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
            &input_buf,
            &output_buf,
            &count_buf,
            &params,
            n,
        );

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        drop(_in_guard);
        drop(_out_guard);
        drop(_ct_guard);

        let n_out = (count_buf.as_slice()[0] as usize).min(n);
        Ok(output_buf.as_slice()[..n_out].to_vec())
    }

    /// Encodes NMS without committing.
    pub fn encode(
        &self,
        ctx: &Context,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        corners: &[CornerPoint],
        config: &NmsConfig,
    ) -> Result<NmsEncodedBuffers> {
        if corners.is_empty() {
            return Err(Error::InvalidConfig(
                "cannot encode NMS with zero corners".into(),
            ));
        }

        let n = corners.len();

        let mut input_buf: UnifiedBuffer<CornerPoint> = UnifiedBuffer::new(ctx.device(), n)?;
        input_buf.write(corners);

        let output_buf: UnifiedBuffer<CornerPoint> = UnifiedBuffer::new(ctx.device(), n)?;

        let mut count_buf: UnifiedBuffer<u32> = UnifiedBuffer::new(ctx.device(), 1)?;
        count_buf.write(&[0u32]);

        let params = NMSParams {
            n_corners: n as u32,
            min_distance: config.min_distance,
        };

        Self::encode_into(
            &self.pipeline,
            encoder,
            &input_buf,
            &output_buf,
            &count_buf,
            &params,
            n,
        );

        Ok(NmsEncodedBuffers {
            input: input_buf,
            output: output_buf,
            count: count_buf,
            n_input: n,
        })
    }

    /// Reads surviving corners after the command buffer has completed.
    pub fn read_results(buffers: &NmsEncodedBuffers) -> Vec<CornerPoint> {
        let n_out = (buffers.count.as_slice()[0] as usize).min(buffers.n_input);
        buffers.output.as_slice()[..n_out].to_vec()
    }

    fn encode_into(
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        input_buf: &UnifiedBuffer<CornerPoint>,
        output_buf: &UnifiedBuffer<CornerPoint>,
        count_buf: &UnifiedBuffer<u32>,
        params: &NMSParams,
        n: usize,
    ) {
        unsafe {
            encoder.setComputePipelineState(pipeline);
            encoder.setBuffer_offset_atIndex(Some(input_buf.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(output_buf.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(count_buf.metal_buffer()), 0, 2);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(params as *const NMSParams as *mut c_void),
                mem::size_of::<NMSParams>(),
                3,
            );

            let tew = pipeline.threadExecutionWidth();
            let grid = MTLSize {
                width: n,
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

/// Buffers returned by [`NmsSuppressor::encode`].
pub struct NmsEncodedBuffers {
    pub input: UnifiedBuffer<CornerPoint>,
    pub output: UnifiedBuffer<CornerPoint>,
    pub count: UnifiedBuffer<u32>,
    pub n_input: usize,
}
