//! GPU color space conversions (RGBA/Gray/HSV).

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
use crate::types::ColorParams;

/// Compiled color conversion pipelines.
pub struct ColorConvert {
    rgba_to_gray_pipe: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    gray_to_rgba_pipe: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    rgba_to_hsv_pipe: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    hsv_to_rgba_pipe: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

unsafe impl Send for ColorConvert {}
unsafe impl Sync for ColorConvert {}

impl ColorConvert {
    pub fn new(ctx: &Context) -> Result<Self> {
        let names = [
            ("rgba_to_gray", "rgba_to_gray"),
            ("gray_to_rgba", "gray_to_rgba"),
            ("rgba_to_hsv", "rgba_to_hsv"),
            ("hsv_to_rgba", "hsv_to_rgba"),
        ];

        let mut pipelines = Vec::new();
        for (name_str, _) in &names {
            let ns_name = objc2_foundation::NSString::from_str(name_str);
            let func = ctx
                .library()
                .newFunctionWithName(&ns_name)
                .ok_or(Error::ShaderMissing((*name_str).into()))?;
            let pipe = ctx
                .device()
                .newComputePipelineStateWithFunction_error(&func)
                .map_err(|e| Error::PipelineCompile(format!("{name_str}: {e}")))?;
            pipelines.push(pipe);
        }

        Ok(Self {
            rgba_to_gray_pipe: pipelines.remove(0),
            gray_to_rgba_pipe: pipelines.remove(0),
            rgba_to_hsv_pipe: pipelines.remove(0),
            hsv_to_rgba_pipe: pipelines.remove(0),
        })
    }

    /// RGBA to grayscale via BT.601 luminance weights.
    pub fn rgba_to_gray(&self, ctx: &Context, input: &Texture, output: &Texture) -> Result<()> {
        self.run_simple(ctx, &self.rgba_to_gray_pipe, input, output)
    }

    /// Grayscale to RGBA (broadcasts gray to RGB, alpha = 1).
    pub fn gray_to_rgba(&self, ctx: &Context, input: &Texture, output: &Texture) -> Result<()> {
        self.run_simple(ctx, &self.gray_to_rgba_pipe, input, output)
    }

    /// RGBA to HSV, packed as R=H, G=S, B=V in an RGBA texture.
    pub fn rgba_to_hsv(&self, ctx: &Context, input: &Texture, output: &Texture) -> Result<()> {
        self.run_simple(ctx, &self.rgba_to_hsv_pipe, input, output)
    }

    /// HSV (packed as R=H, G=S, B=V) back to RGBA.
    pub fn hsv_to_rgba(&self, ctx: &Context, input: &Texture, output: &Texture) -> Result<()> {
        self.run_simple(ctx, &self.hsv_to_rgba_pipe, input, output)
    }

    /// Encodes RGBA-to-gray without committing.
    pub fn encode_rgba_to_gray(
        &self,
        cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
        input: &Texture,
        output: &Texture,
    ) -> Result<()> {
        self.encode_simple(cmd_buf, &self.rgba_to_gray_pipe, input, output)
    }

    /// Encodes gray-to-RGBA without committing.
    pub fn encode_gray_to_rgba(
        &self,
        cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
        input: &Texture,
        output: &Texture,
    ) -> Result<()> {
        self.encode_simple(cmd_buf, &self.gray_to_rgba_pipe, input, output)
    }

    /// Encodes RGBA-to-HSV without committing.
    pub fn encode_rgba_to_hsv(
        &self,
        cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
        input: &Texture,
        output: &Texture,
    ) -> Result<()> {
        self.encode_simple(cmd_buf, &self.rgba_to_hsv_pipe, input, output)
    }

    /// Encodes HSV-to-RGBA without committing.
    pub fn encode_hsv_to_rgba(
        &self,
        cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
        input: &Texture,
        output: &Texture,
    ) -> Result<()> {
        self.encode_simple(cmd_buf, &self.hsv_to_rgba_pipe, input, output)
    }

    fn encode_simple(
        &self,
        cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        input: &Texture,
        output: &Texture,
    ) -> Result<()> {
        let w = input.width();
        let h = input.height();
        let params = ColorParams {
            width: w,
            height: h,
        };

        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or(Error::Gpu("failed to create compute encoder".into()))?;

        Self::dispatch(&encoder, pipeline, input, output, &params, w, h);

        encoder.endEncoding();
        Ok(())
    }

    fn run_simple(
        &self,
        ctx: &Context,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        input: &Texture,
        output: &Texture,
    ) -> Result<()> {
        let w = input.width();
        let h = input.height();
        let params = ColorParams {
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

        Self::dispatch(&encoder, pipeline, input, output, &params, w, h);

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    fn dispatch(
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        input: &Texture,
        output: &Texture,
        params: &ColorParams,
        w: u32,
        h: u32,
    ) {
        unsafe {
            encoder.setComputePipelineState(pipeline);
            encoder.setTexture_atIndex(Some(input.raw()), 0);
            encoder.setTexture_atIndex(Some(output.raw()), 1);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(params as *const ColorParams as *mut c_void),
                mem::size_of::<ColorParams>(),
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
