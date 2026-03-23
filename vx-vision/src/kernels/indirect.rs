//! Indirect dispatch for GPU-driven kernel chaining (e.g. FAST -> Harris).
//!
//! Eliminates the CPU round-trip for reading FAST's atomic corner count by
//! computing Metal indirect dispatch arguments on the GPU.

use core::ffi::c_void;
use core::ptr::NonNull;
use std::mem;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLLibrary, MTLSize,
};

use vx_gpu::UnifiedBuffer;
use crate::context::Context;
use crate::error::{Error, Result};
use crate::texture::Texture;
use crate::types::{CornerPoint, HarrisParams, IndirectArgs, IndirectSetupParams};

/// GPU-side indirect dispatch for chaining FAST -> Harris without CPU readback.
pub struct IndirectDispatch {
    setup_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    harris_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

unsafe impl Send for IndirectDispatch {}
unsafe impl Sync for IndirectDispatch {}

/// Must outlive the command buffer.
pub struct IndirectArgsBuffer {
    pub args: UnifiedBuffer<IndirectArgs>,
}

impl IndirectDispatch {
    /// Compiles both pipelines.
    pub fn new(ctx: &Context) -> Result<Self> {
        let setup_name = objc2_foundation::ns_string!("prepare_indirect_args");
        let harris_name = objc2_foundation::ns_string!("harris_response");

        let setup_func = ctx.library().newFunctionWithName(setup_name)
            .ok_or(Error::ShaderMissing("prepare_indirect_args".into()))?;
        let harris_func = ctx.library().newFunctionWithName(harris_name)
            .ok_or(Error::ShaderMissing("harris_response".into()))?;

        let setup_pipeline = ctx.device()
            .newComputePipelineStateWithFunction_error(&setup_func)
            .map_err(|e| Error::PipelineCompile(format!("prepare_indirect_args: {e}")))?;
        let harris_pipeline = ctx.device()
            .newComputePipelineStateWithFunction_error(&harris_func)
            .map_err(|e| Error::PipelineCompile(format!("harris_response: {e}")))?;

        Ok(Self { setup_pipeline, harris_pipeline })
    }

    /// Encodes indirect args from FAST's atomic counter. Must run between FAST and Harris.
    pub fn encode_indirect_args(
        &self,
        ctx: &Context,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        count_buf: &UnifiedBuffer<u32>,
    ) -> Result<UnifiedBuffer<IndirectArgs>> {
        let args_buf: UnifiedBuffer<IndirectArgs> =
            UnifiedBuffer::new(ctx.device(), 1)?;

        let harris_tew = self.harris_pipeline.threadExecutionWidth() as u32;
        let params = IndirectSetupParams {
            threads_per_threadgroup: harris_tew,
        };

        unsafe {
            encoder.setComputePipelineState(&self.setup_pipeline);
            encoder.setBuffer_offset_atIndex(Some(count_buf.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(args_buf.metal_buffer()), 0, 1);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(
                    &params as *const IndirectSetupParams as *mut c_void,
                ),
                mem::size_of::<IndirectSetupParams>(),
                2,
            );

            let grid = MTLSize { width: 1, height: 1, depth: 1 };
            let tg = MTLSize { width: 1, height: 1, depth: 1 };
            encoder.dispatchThreads_threadsPerThreadgroup(grid, tg);
        }

        Ok(args_buf)
    }

    /// Encodes Harris scoring with indirect dispatch from the GPU-computed args buffer.
    pub fn encode_harris_indirect(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        texture: &Texture,
        corner_buf: &UnifiedBuffer<CornerPoint>,
        args_buf: &UnifiedBuffer<IndirectArgs>,
        config: &crate::kernels::harris::HarrisConfig,
        max_corners: u32,
    ) -> Result<()> {
        let params = HarrisParams {
            n_corners: max_corners, // GPU reads actual count; shader bounds-checks via n_corners
            patch_radius: config.patch_radius,
            k: config.k,
        };

        unsafe {
            encoder.setComputePipelineState(&self.harris_pipeline);
            encoder.setTexture_atIndex(Some(texture.raw()), 0);
            encoder.setBuffer_offset_atIndex(Some(corner_buf.metal_buffer()), 0, 0);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(
                    &params as *const HarrisParams as *mut c_void,
                ),
                mem::size_of::<HarrisParams>(),
                1,
            );

            let tew = self.harris_pipeline.threadExecutionWidth();
            let tg = MTLSize { width: tew, height: 1, depth: 1 };

            // objc2-metal lacks this binding; use raw msg_send
            let indirect_buffer = args_buf.metal_buffer();
            let offset: usize = 0;
            let _: () = objc2::msg_send![
                encoder,
                dispatchThreadgroupsWithIndirectBuffer: indirect_buffer,
                indirectBufferOffset: offset,
                threadsPerThreadgroup: tg
            ];
        }

        Ok(())
    }

    /// Encodes indirect args + Harris after FAST has been encoded into `encoder`.
    pub fn encode_fast_harris_chain(
        &self,
        ctx: &Context,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        texture: &Texture,
        fast_buffers: &crate::kernels::fast::EncodedBuffers,
        harris_config: &crate::kernels::harris::HarrisConfig,
    ) -> Result<IndirectArgsBuffer> {
        let args_buf = self.encode_indirect_args(ctx, encoder, &fast_buffers.count)?;

        self.encode_harris_indirect(
            encoder, texture,
            &fast_buffers.corners, &args_buf,
            harris_config, fast_buffers.max_corners,
        )?;

        Ok(IndirectArgsBuffer { args: args_buf })
    }
}
