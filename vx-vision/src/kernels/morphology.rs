//! Morphological operations (erode, dilate, open, close) with rectangular structuring elements.

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
use crate::types::MorphParams;

/// Configuration for morphological operations.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct MorphConfig {
    /// Half-width of the structuring element in x.
    pub radius_x: i32,

    /// Half-height of the structuring element in y.
    pub radius_y: i32,
}

impl MorphConfig {
    /// Creates a config with the given structuring element radii.
    pub fn new(radius_x: i32, radius_y: i32) -> Self {
        Self { radius_x, radius_y }
    }
}

impl Default for MorphConfig {
    fn default() -> Self {
        Self {
            radius_x: 1,
            radius_y: 1,
        }
    }
}

/// Keeps the intermediate texture alive for compound morphological operations.
pub struct MorphEncodedState {
    pub _intermediate: Texture,
}

/// Compiled erode/dilate pipelines.
pub struct Morphology {
    erode_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    dilate_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

unsafe impl Send for Morphology {}
unsafe impl Sync for Morphology {}

impl Morphology {
    pub fn new(ctx: &Context) -> Result<Self> {
        let erode_name = objc2_foundation::ns_string!("erode");
        let dilate_name = objc2_foundation::ns_string!("dilate");

        let erode_func = ctx
            .library()
            .newFunctionWithName(erode_name)
            .ok_or(Error::ShaderMissing("erode".into()))?;
        let dilate_func = ctx
            .library()
            .newFunctionWithName(dilate_name)
            .ok_or(Error::ShaderMissing("dilate".into()))?;

        let erode_pipeline = ctx
            .device()
            .newComputePipelineStateWithFunction_error(&erode_func)
            .map_err(|e| Error::PipelineCompile(format!("erode: {e}")))?;
        let dilate_pipeline = ctx
            .device()
            .newComputePipelineStateWithFunction_error(&dilate_func)
            .map_err(|e| Error::PipelineCompile(format!("dilate: {e}")))?;

        Ok(Self {
            erode_pipeline,
            dilate_pipeline,
        })
    }

    /// Minimum filter over the structuring element neighborhood.
    pub fn erode(
        &self,
        ctx: &Context,
        input: &Texture,
        output: &Texture,
        config: &MorphConfig,
    ) -> Result<()> {
        self.run_op(ctx, &self.erode_pipeline, input, output, config)
    }

    /// Maximum filter over the structuring element neighborhood.
    pub fn dilate(
        &self,
        ctx: &Context,
        input: &Texture,
        output: &Texture,
        config: &MorphConfig,
    ) -> Result<()> {
        self.run_op(ctx, &self.dilate_pipeline, input, output, config)
    }

    /// Morphological opening (erode then dilate). Removes small bright regions.
    pub fn open(
        &self,
        ctx: &Context,
        input: &Texture,
        output: &Texture,
        config: &MorphConfig,
    ) -> Result<()> {
        let w = input.width();
        let h = input.height();
        let intermediate = Texture::output_gray8(ctx.device(), w, h)?;
        self.run_op(ctx, &self.erode_pipeline, input, &intermediate, config)?;
        self.run_op(ctx, &self.dilate_pipeline, &intermediate, output, config)?;
        Ok(())
    }

    /// Morphological closing (dilate then erode). Fills small dark regions.
    pub fn close(
        &self,
        ctx: &Context,
        input: &Texture,
        output: &Texture,
        config: &MorphConfig,
    ) -> Result<()> {
        let w = input.width();
        let h = input.height();
        let intermediate = Texture::output_gray8(ctx.device(), w, h)?;
        self.run_op(ctx, &self.dilate_pipeline, input, &intermediate, config)?;
        self.run_op(ctx, &self.erode_pipeline, &intermediate, output, config)?;
        Ok(())
    }

    /// Encodes erode without committing.
    pub fn encode_erode(
        &self,
        cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
        input: &Texture,
        output: &Texture,
        config: &MorphConfig,
    ) -> Result<()> {
        self.encode_op(cmd_buf, &self.erode_pipeline, input, output, config)
    }

    /// Encodes dilate without committing.
    pub fn encode_dilate(
        &self,
        cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
        input: &Texture,
        output: &Texture,
        config: &MorphConfig,
    ) -> Result<()> {
        self.encode_op(cmd_buf, &self.dilate_pipeline, input, output, config)
    }

    /// Encodes opening (erode then dilate) without committing.
    pub fn encode_open(
        &self,
        ctx: &Context,
        cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
        input: &Texture,
        output: &Texture,
        config: &MorphConfig,
    ) -> Result<MorphEncodedState> {
        let w = input.width();
        let h = input.height();
        let intermediate = Texture::output_gray8(ctx.device(), w, h)?;
        self.encode_op(cmd_buf, &self.erode_pipeline, input, &intermediate, config)?;
        self.encode_op(
            cmd_buf,
            &self.dilate_pipeline,
            &intermediate,
            output,
            config,
        )?;
        Ok(MorphEncodedState {
            _intermediate: intermediate,
        })
    }

    /// Encodes closing (dilate then erode) without committing.
    pub fn encode_close(
        &self,
        ctx: &Context,
        cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
        input: &Texture,
        output: &Texture,
        config: &MorphConfig,
    ) -> Result<MorphEncodedState> {
        let w = input.width();
        let h = input.height();
        let intermediate = Texture::output_gray8(ctx.device(), w, h)?;
        self.encode_op(cmd_buf, &self.dilate_pipeline, input, &intermediate, config)?;
        self.encode_op(cmd_buf, &self.erode_pipeline, &intermediate, output, config)?;
        Ok(MorphEncodedState {
            _intermediate: intermediate,
        })
    }

    fn encode_op(
        &self,
        cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        input: &Texture,
        output: &Texture,
        config: &MorphConfig,
    ) -> Result<()> {
        let w = input.width();
        let h = input.height();
        let params = MorphParams {
            width: w,
            height: h,
            radius_x: config.radius_x,
            radius_y: config.radius_y,
        };

        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or(Error::Gpu("failed to create compute encoder".into()))?;

        Self::dispatch(&encoder, pipeline, &params, input, output, w, h);

        encoder.endEncoding();
        Ok(())
    }

    fn run_op(
        &self,
        ctx: &Context,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        input: &Texture,
        output: &Texture,
        config: &MorphConfig,
    ) -> Result<()> {
        let w = input.width();
        let h = input.height();
        let params = MorphParams {
            width: w,
            height: h,
            radius_x: config.radius_x,
            radius_y: config.radius_y,
        };

        let cmd_buf = ctx
            .queue()
            .commandBuffer()
            .ok_or(Error::Gpu("failed to create command buffer".into()))?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or(Error::Gpu("failed to create compute encoder".into()))?;

        Self::dispatch(&encoder, pipeline, &params, input, output, w, h);

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    fn dispatch(
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        params: &MorphParams,
        input: &Texture,
        output: &Texture,
        w: u32,
        h: u32,
    ) {
        unsafe {
            encoder.setComputePipelineState(pipeline);
            encoder.setTexture_atIndex(Some(input.raw()), 0);
            encoder.setTexture_atIndex(Some(output.raw()), 1);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(params as *const MorphParams as *mut c_void),
                mem::size_of::<MorphParams>(),
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
