//! Template matching via normalized cross-correlation.

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
use crate::types::TemplateParams;

/// Template matching result.
pub struct TemplateMatchResult {
    /// NCC correlation map; higher values indicate better matches.
    pub correlation: Texture,
    /// Best-match location (x, y).
    pub best_x: u32,
    pub best_y: u32,
    /// NCC score at the best-match location.
    pub best_score: f32,
}

/// NCC template matching compute pipeline.
pub struct TemplateMatcher {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl TemplateMatcher {
    pub fn new(ctx: &Context) -> Result<Self, String> {
        let name = objc2_foundation::ns_string!("template_match_ncc");
        let func = ctx.library().newFunctionWithName(name)
            .ok_or_else(|| "Missing kernel function 'template_match_ncc'".to_string())?;
        let pipeline = ctx.device()
            .newComputePipelineStateWithFunction_error(&func)
            .map_err(|e| format!("Failed to create template_match_ncc pipeline: {e}"))?;
        Ok(Self { pipeline })
    }

    /// Matches a template against a grayscale image using NCC.
    ///
    /// Returns the correlation map and best-match location/score.
    pub fn match_template(
        &self,
        ctx:      &Context,
        image:    &Texture,
        template: &Texture,
    ) -> Result<TemplateMatchResult, String> {
        let img_w = image.width();
        let img_h = image.height();
        let tpl_w = template.width();
        let tpl_h = template.height();

        if tpl_w > img_w || tpl_h > img_h {
            return Err("Template is larger than image".to_string());
        }

        let out_w = img_w - tpl_w + 1;
        let out_h = img_h - tpl_h + 1;

        // Precompute template statistics on CPU
        let tpl_pixels = template.read_gray8();
        let n = (tpl_w * tpl_h) as f32;
        let tpl_mean: f32 = tpl_pixels.iter().map(|&p| p as f32 / 255.0).sum::<f32>() / n;
        let tpl_norm: f32 = tpl_pixels.iter()
            .map(|&p| {
                let d = p as f32 / 255.0 - tpl_mean;
                d * d
            })
            .sum::<f32>()
            .sqrt();

        let params = TemplateParams {
            img_width:  img_w,
            img_height: img_h,
            tpl_width:  tpl_w,
            tpl_height: tpl_h,
            tpl_mean,
            tpl_norm,
        };

        let result_tex = Texture::output_r32float(ctx.device(), out_w, out_h)?;

        let cmd_buf = ctx.queue().commandBuffer()
            .ok_or("Failed to create command buffer")?;
        let encoder = cmd_buf.computeCommandEncoder()
            .ok_or("Failed to create compute encoder")?;

        unsafe {
            encoder.setComputePipelineState(&self.pipeline);
            encoder.setTexture_atIndex(Some(image.raw()),      0);
            encoder.setTexture_atIndex(Some(template.raw()),   1);
            encoder.setTexture_atIndex(Some(result_tex.raw()), 2);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(&params as *const TemplateParams as *mut c_void),
                mem::size_of::<TemplateParams>(),
                0,
            );

            let tew    = self.pipeline.threadExecutionWidth();
            let max_tg = self.pipeline.maxTotalThreadsPerThreadgroup();
            let tg_h   = (max_tg / tew).max(1);
            let grid    = MTLSize { width: out_w as usize, height: out_h as usize, depth: 1 };
            let tg_size = MTLSize { width: tew,            height: tg_h,           depth: 1 };
            encoder.dispatchThreads_threadsPerThreadgroup(grid, tg_size);
        }

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        // Scan correlation map for peak
        let corr_data = result_tex.read_r32float();
        let mut best_score = f32::NEG_INFINITY;
        let mut best_idx = 0usize;
        for (i, &val) in corr_data.iter().enumerate() {
            if val > best_score {
                best_score = val;
                best_idx = i;
            }
        }
        let best_x = (best_idx % out_w as usize) as u32;
        let best_y = (best_idx / out_w as usize) as u32;

        Ok(TemplateMatchResult {
            correlation: result_tex,
            best_x,
            best_y,
            best_score,
        })
    }
}
