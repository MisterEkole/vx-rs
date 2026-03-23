//! Depth map colorization for visualization.

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
use crate::texture::Texture;
use crate::types::DepthColorizeParams;

#[cfg(feature = "reconstruction")]
use crate::types_3d::Colormap;

/// Configuration for depth colorization.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct DepthColorizeConfig {
    /// Minimum depth for colormap range.
    pub min_depth: f32,
    /// Maximum depth for colormap range.
    pub max_depth: f32,
    /// Colormap to use (0 = Turbo, 1 = Jet, 2 = Inferno).
    pub colormap_id: u32,
}

impl Default for DepthColorizeConfig {
    fn default() -> Self {
        Self {
            min_depth: 0.1,
            max_depth: 10.0,
            colormap_id: 0,
        }
    }
}

impl DepthColorizeConfig {
    pub fn new(min_depth: f32, max_depth: f32) -> Self {
        Self {
            min_depth,
            max_depth,
            ..Default::default()
        }
    }

    /// Creates a config with a specific colormap.
    #[cfg(feature = "reconstruction")]
    pub fn with_colormap(min_depth: f32, max_depth: f32, colormap: Colormap) -> Self {
        let colormap_id = match colormap {
            Colormap::Turbo => 0,
            Colormap::Jet => 1,
            Colormap::Inferno => 2,
        };
        Self {
            min_depth,
            max_depth,
            colormap_id,
        }
    }
}

/// Compiled depth colorization pipeline.
pub struct DepthColorize {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

unsafe impl Send for DepthColorize {}
unsafe impl Sync for DepthColorize {}

impl DepthColorize {
    pub fn new(ctx: &Context) -> Result<Self> {
        let func = ctx
            .library()
            .newFunctionWithName(objc2_foundation::ns_string!("depth_colorize"))
            .ok_or(Error::ShaderMissing("depth_colorize".into()))?;
        let pipeline = ctx
            .device()
            .newComputePipelineStateWithFunction_error(&func)
            .map_err(|e| Error::PipelineCompile(format!("depth_colorize: {e}")))?;
        Ok(Self { pipeline })
    }

    /// Colorizes a depth map (R32Float → RGBA8Unorm). Synchronous.
    pub fn apply(
        &self,
        ctx: &Context,
        input: &Texture,
        output: &Texture,
        config: &DepthColorizeConfig,
    ) -> Result<()> {
        let w = input.width();
        let h = input.height();
        let params = DepthColorizeParams {
            min_depth: config.min_depth,
            max_depth: config.max_depth,
            colormap_id: config.colormap_id,
            width: w,
            height: h,
        };

        let cmd_buf = ctx
            .queue()
            .commandBuffer()
            .ok_or(Error::Gpu("failed to create command buffer".into()))?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or(Error::Gpu("failed to create compute encoder".into()))?;

        Self::encode_pass(&self.pipeline, &encoder, input, output, &params, w, h);

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    /// Encodes depth colorization without committing.
    pub fn encode(
        &self,
        cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
        input: &Texture,
        output: &Texture,
        config: &DepthColorizeConfig,
    ) -> Result<()> {
        let w = input.width();
        let h = input.height();
        let params = DepthColorizeParams {
            min_depth: config.min_depth,
            max_depth: config.max_depth,
            colormap_id: config.colormap_id,
            width: w,
            height: h,
        };

        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or(Error::Gpu("failed to create compute encoder".into()))?;
        Self::encode_pass(&self.pipeline, &encoder, input, output, &params, w, h);
        encoder.endEncoding();
        Ok(())
    }

    fn encode_pass(
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        input: &Texture,
        output: &Texture,
        params: &DepthColorizeParams,
        w: u32,
        h: u32,
    ) {
        unsafe {
            encoder.setComputePipelineState(pipeline);
            encoder.setTexture_atIndex(Some(input.raw()), 0);
            encoder.setTexture_atIndex(Some(output.raw()), 1);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(params as *const DepthColorizeParams as *mut c_void),
                mem::size_of::<DepthColorizeParams>(),
                0,
            );

            let tew = pipeline.threadExecutionWidth();
            let max_tg = pipeline.maxTotalThreadsPerThreadgroup();
            let tg_h = (max_tg / tew).max(1);
            let grid = MTLSize {
                width: w as usize,
                height: h as usize,
                depth: 1,
            };
            let tg_size = MTLSize {
                width: tew,
                height: tg_h,
                depth: 1,
            };
            encoder.dispatchThreads_threadsPerThreadgroup(grid, tg_size);
        }
    }
}
