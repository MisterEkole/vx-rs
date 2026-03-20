//! Morphological operations (erode, dilate, open, close) with rectangular structuring elements.

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

use crate::context::Context;
use crate::texture::Texture;
use crate::types::MorphParams;

/// Configuration for morphological operations.
#[derive(Clone, Debug)]
pub struct MorphConfig {
    /// Half-width of the structuring element in x.
    pub radius_x: i32,

    /// Half-height of the structuring element in y.
    pub radius_y: i32,
}

impl Default for MorphConfig {
    fn default() -> Self {
        Self { radius_x: 1, radius_y: 1 }
    }
}

/// Compiled morphological operation pipelines.
pub struct Morphology {
    erode_pipeline:  Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    dilate_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl Morphology {
    pub fn new(ctx: &Context) -> Result<Self, String> {
        let erode_name  = objc2_foundation::ns_string!("erode");
        let dilate_name = objc2_foundation::ns_string!("dilate");

        let erode_func = ctx.library().newFunctionWithName(erode_name)
            .ok_or_else(|| "Missing kernel function 'erode'".to_string())?;
        let dilate_func = ctx.library().newFunctionWithName(dilate_name)
            .ok_or_else(|| "Missing kernel function 'dilate'".to_string())?;

        let erode_pipeline = ctx.device()
            .newComputePipelineStateWithFunction_error(&erode_func)
            .map_err(|e| format!("Failed to create erode pipeline: {e}"))?;
        let dilate_pipeline = ctx.device()
            .newComputePipelineStateWithFunction_error(&dilate_func)
            .map_err(|e| format!("Failed to create dilate pipeline: {e}"))?;

        Ok(Self { erode_pipeline, dilate_pipeline })
    }

    /// Minimum filter over the structuring element neighborhood.
    pub fn erode(
        &self,
        ctx:    &Context,
        input:  &Texture,
        output: &Texture,
        config: &MorphConfig,
    ) -> Result<(), String> {
        self.run_op(ctx, &self.erode_pipeline, input, output, config)
    }

    /// Maximum filter over the structuring element neighborhood.
    pub fn dilate(
        &self,
        ctx:    &Context,
        input:  &Texture,
        output: &Texture,
        config: &MorphConfig,
    ) -> Result<(), String> {
        self.run_op(ctx, &self.dilate_pipeline, input, output, config)
    }

    /// Morphological opening (erode then dilate). Removes small bright regions.
    pub fn open(
        &self,
        ctx:    &Context,
        input:  &Texture,
        output: &Texture,
        config: &MorphConfig,
    ) -> Result<(), String> {
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
        ctx:    &Context,
        input:  &Texture,
        output: &Texture,
        config: &MorphConfig,
    ) -> Result<(), String> {
        let w = input.width();
        let h = input.height();
        let intermediate = Texture::output_gray8(ctx.device(), w, h)?;
        self.run_op(ctx, &self.dilate_pipeline, input, &intermediate, config)?;
        self.run_op(ctx, &self.erode_pipeline, &intermediate, output, config)?;
        Ok(())
    }

    fn run_op(
        &self,
        ctx:      &Context,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        input:    &Texture,
        output:   &Texture,
        config:   &MorphConfig,
    ) -> Result<(), String> {
        let w = input.width();
        let h = input.height();
        let params = MorphParams {
            width:    w,
            height:   h,
            radius_x: config.radius_x,
            radius_y: config.radius_y,
        };

        let cmd_buf = ctx.queue().commandBuffer()
            .ok_or("Failed to create command buffer")?;
        let encoder = cmd_buf.computeCommandEncoder()
            .ok_or("Failed to create compute encoder")?;

        unsafe {
            encoder.setComputePipelineState(pipeline);
            encoder.setTexture_atIndex(Some(input.raw()),  0);
            encoder.setTexture_atIndex(Some(output.raw()), 1);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(&params as *const MorphParams as *mut c_void),
                mem::size_of::<MorphParams>(),
                0,
            );

            let tew    = pipeline.threadExecutionWidth();
            let max_tg = pipeline.maxTotalThreadsPerThreadgroup();
            let tg_h   = (max_tg / tew).max(1);
            let grid    = MTLSize { width: w as usize, height: h as usize, depth: 1 };
            let tg_size = MTLSize { width: tew,        height: tg_h,       depth: 1 };
            encoder.dispatchThreads_threadsPerThreadgroup(grid, tg_size);
        }

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }
}
