//! Canny edge detection pipeline (blur -> Sobel -> NMS -> hysteresis).

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
use crate::error::{Error, Result};
use crate::kernels::gaussian::{GaussianBlur, GaussianConfig, GaussianEncodedState};
use crate::kernels::sobel::{SobelFilter, SobelEncodedState};
use crate::texture::Texture;
use crate::types::CannyParams;

/// Configuration for Canny edge detection.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct CannyConfig {
    /// Low hysteresis threshold. Edges below this are discarded.
    pub low_threshold: f32,

    /// High hysteresis threshold. Edges above this are always kept.
    pub high_threshold: f32,

    /// Gaussian blur sigma applied before gradient computation.
    pub blur_sigma: f32,

    /// Gaussian blur radius. Rule of thumb: `ceil(3 * sigma)`.
    pub blur_radius: u32,
}

impl Default for CannyConfig {
    fn default() -> Self {
        Self {
            low_threshold:  0.05,
            high_threshold: 0.15,
            blur_sigma:     1.4,
            blur_radius:    4,
        }
    }
}

/// Keeps textures alive until the command buffer completes.
pub struct CannyEncodedState {
    /// Final edge map (R32Float, 1.0 = edge).
    pub edges: Texture,
    _blurred: Texture,
    _sobel_state: SobelEncodedState,
    _nms_output: Texture,
    _blur_state: GaussianEncodedState,
}

/// Compiled Canny edge detection pipeline.
pub struct CannyDetector {
    blur:      GaussianBlur,
    sobel:     SobelFilter,
    nms_pipe:  Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    hyst_pipe: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl CannyDetector {
    pub fn new(ctx: &Context) -> Result<Self> {
        let blur  = GaussianBlur::new(ctx)?;
        let sobel = SobelFilter::new(ctx)?;

        let nms_name  = objc2_foundation::ns_string!("canny_nms");
        let hyst_name = objc2_foundation::ns_string!("canny_hysteresis");

        let nms_func = ctx.library().newFunctionWithName(nms_name)
            .ok_or_else(|| Error::ShaderMissing("canny_nms".into()))?;
        let hyst_func = ctx.library().newFunctionWithName(hyst_name)
            .ok_or_else(|| Error::ShaderMissing("canny_hysteresis".into()))?;

        let nms_pipe = ctx.device()
            .newComputePipelineStateWithFunction_error(&nms_func)
            .map_err(|e| Error::PipelineCompile(format!("canny_nms: {e}")))?;
        let hyst_pipe = ctx.device()
            .newComputePipelineStateWithFunction_error(&hyst_func)
            .map_err(|e| Error::PipelineCompile(format!("canny_hysteresis: {e}")))?;

        Ok(Self { blur, sobel, nms_pipe, hyst_pipe })
    }

    /// Runs the full Canny pipeline. Returns R32Float (1.0 = edge).
    pub fn detect(
        &self,
        ctx:    &Context,
        input:  &Texture,
        config: &CannyConfig,
    ) -> Result<Texture> {
        let w = input.width();
        let h = input.height();

        let blurred = ctx.texture_output_gray8(w, h)?;
        self.blur.apply(ctx, input, &blurred, &GaussianConfig {
            sigma:  config.blur_sigma,
            radius: config.blur_radius,
        })?;

        let sobel_result = self.sobel.compute(ctx, &blurred)?;

        let nms_output = Texture::intermediate_r32float(ctx.device(), w, h)?;

        let params = CannyParams {
            width:          w,
            height:         h,
            low_threshold:  config.low_threshold,
            high_threshold: config.high_threshold,
        };

        let cmd_buf = ctx.queue().commandBuffer()
            .ok_or(Error::Gpu("failed to create command buffer".into()))?;

        {
            let encoder = cmd_buf.computeCommandEncoder()
                .ok_or(Error::Gpu("failed to create compute encoder".into()))?;
            Self::encode_2d(&self.nms_pipe, &encoder, &[
                &sobel_result.magnitude, &sobel_result.direction, &nms_output,
            ], &params, w, h);
            encoder.endEncoding();
        }

        let edges = Texture::intermediate_r32float(ctx.device(), w, h)?;
        {
            let encoder = cmd_buf.computeCommandEncoder()
                .ok_or(Error::Gpu("failed to create compute encoder".into()))?;
            Self::encode_2d(&self.hyst_pipe, &encoder, &[
                &nms_output, &edges,
            ], &params, w, h);
            encoder.endEncoding();
        }

        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        Ok(edges)
    }

    /// Encodes the full Canny pipeline into `cmd_buf` without committing.
    pub fn encode(
        &self,
        ctx:     &Context,
        cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
        input:   &Texture,
        config:  &CannyConfig,
    ) -> Result<CannyEncodedState> {
        let w = input.width();
        let h = input.height();

        let blurred = Texture::intermediate_gray8(ctx.device(), w, h)?;
        let blur_state = self.blur.encode(ctx, cmd_buf, input, &blurred, &GaussianConfig {
            sigma:  config.blur_sigma,
            radius: config.blur_radius,
        })?;

        let sobel_state = self.sobel.encode(ctx, cmd_buf, &blurred)?;

        let nms_output = Texture::intermediate_r32float(ctx.device(), w, h)?;
        let params = CannyParams {
            width:          w,
            height:         h,
            low_threshold:  config.low_threshold,
            high_threshold: config.high_threshold,
        };

        {
            let encoder = cmd_buf.computeCommandEncoder()
                .ok_or(Error::Gpu("failed to create compute encoder".into()))?;
            Self::encode_2d(&self.nms_pipe, &encoder, &[
                &sobel_state.magnitude, &sobel_state.direction, &nms_output,
            ], &params, w, h);
            encoder.endEncoding();
        }

        let edges = Texture::intermediate_r32float(ctx.device(), w, h)?;
        {
            let encoder = cmd_buf.computeCommandEncoder()
                .ok_or(Error::Gpu("failed to create compute encoder".into()))?;
            Self::encode_2d(&self.hyst_pipe, &encoder, &[
                &nms_output, &edges,
            ], &params, w, h);
            encoder.endEncoding();
        }

        Ok(CannyEncodedState {
            edges,
            _blurred: blurred,
            _sobel_state: sobel_state,
            _nms_output: nms_output,
            _blur_state: blur_state,
        })
    }

    fn encode_2d(
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        encoder:  &ProtocolObject<dyn MTLComputeCommandEncoder>,
        textures: &[&Texture],
        params:   &CannyParams,
        width:    u32,
        height:   u32,
    ) {
        unsafe {
            encoder.setComputePipelineState(pipeline);
            for (i, tex) in textures.iter().enumerate() {
                encoder.setTexture_atIndex(Some(tex.raw()), i);
            }
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(params as *const CannyParams as *mut c_void),
                mem::size_of::<CannyParams>(),
                0,
            );

            let tew    = pipeline.threadExecutionWidth();
            let max_tg = pipeline.maxTotalThreadsPerThreadgroup();
            let tg_h   = (max_tg / tew).max(1);
            let grid    = MTLSize { width: width as usize,  height: height as usize, depth: 1 };
            let tg_size = MTLSize { width: tew,             height: tg_h,            depth: 1 };
            encoder.dispatchThreads_threadsPerThreadgroup(grid, tg_size);
        }
    }
}

unsafe impl Send for CannyDetector {}
unsafe impl Sync for CannyDetector {}
