//! Lens undistortion via precomputed per-pixel coordinate maps (OpenCV-style remap).

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice,
    MTLLibrary, MTLSize,
};

use crate::context::Context;
use crate::error::{Error, Result};
use crate::texture::Texture;

/// Compiled lens undistortion pipeline.
pub struct Undistorter {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl Undistorter {
    /// Compiles the undistortion pipeline.
    pub fn new(ctx: &Context) -> Result<Self> {
        let name = objc2_foundation::ns_string!("undistort");

        let func = ctx.library().newFunctionWithName(name)
            .ok_or(Error::ShaderMissing("undistort".into()))?;

        let pipeline = ctx.device().newComputePipelineStateWithFunction_error(&func)
            .map_err(|e| Error::PipelineCompile(format!("undistort: {e}")))?;

        Ok(Self { pipeline })
    }

    /// Remaps `input` into `output` using per-pixel coordinate maps. Synchronous.
    pub fn apply(
        &self,
        ctx: &Context,
        input:  &Texture,
        map_x:  &Texture,
        map_y:  &Texture,
        output: &Texture,
    ) -> Result<()> {
        let width  = output.width();
        let height = output.height();

        let cmd_buf = ctx.queue().commandBuffer()
            .ok_or(Error::Gpu("failed to create command buffer".into()))?;
        let encoder = cmd_buf.computeCommandEncoder()
            .ok_or(Error::Gpu("failed to create compute encoder".into()))?;

        Self::encode_into(&self.pipeline, &encoder, input, map_x, map_y, output, width, height);

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        Ok(())
    }

    /// Encodes the undistortion without committing.
    pub fn encode(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        input:   &Texture,
        map_x:   &Texture,
        map_y:   &Texture,
        output:  &Texture,
    ) {
        let width  = output.width();
        let height = output.height();
        Self::encode_into(&self.pipeline, encoder, input, map_x, map_y, output, width, height);
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_into(
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        encoder:  &ProtocolObject<dyn MTLComputeCommandEncoder>,
        input:    &Texture,
        map_x:    &Texture,
        map_y:    &Texture,
        output:   &Texture,
        width:    u32,
        height:   u32,
    ) {
        unsafe {
            encoder.setComputePipelineState(pipeline);
            encoder.setTexture_atIndex(Some(input.raw()),  0);
            encoder.setTexture_atIndex(Some(map_x.raw()),  1);
            encoder.setTexture_atIndex(Some(map_y.raw()),  2);
            encoder.setTexture_atIndex(Some(output.raw()), 3);

            let tew    = pipeline.threadExecutionWidth();
            let max_tg = pipeline.maxTotalThreadsPerThreadgroup();
            let tg_h   = (max_tg / tew).max(1);

            let grid    = MTLSize { width: width as usize,  height: height as usize, depth: 1 };
            let tg_size = MTLSize { width: tew,             height: tg_h,            depth: 1 };

            encoder.dispatchThreads_threadsPerThreadgroup(grid, tg_size);
        }
    }
}

unsafe impl Send for Undistorter {}
unsafe impl Sync for Undistorter {}
