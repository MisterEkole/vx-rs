//! Horn-Schunck dense optical flow.

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
use crate::types::FlowParams;

/// Configuration for Horn-Schunck dense optical flow.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct DenseFlowConfig {
    /// Smoothness weight. Higher values produce smoother flow.
    pub alpha: f32,

    /// Jacobi solver iterations.
    pub iterations: u32,
}

impl Default for DenseFlowConfig {
    fn default() -> Self {
        Self {
            alpha: 10.0,
            iterations: 100,
        }
    }
}

/// Horn-Schunck dense optical flow compute pipelines.
pub struct DenseFlow {
    deriv_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    iter_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

/// Dense optical flow result.
pub struct DenseFlowResult {
    /// Horizontal flow (R32Float).
    pub flow_u: Texture,
    /// Vertical flow (R32Float).
    pub flow_v: Texture,
}

/// Keeps textures alive until the command buffer completes.
pub struct DenseFlowEncodedState {
    /// Horizontal flow (R32Float).
    pub flow_u: Texture,
    /// Vertical flow (R32Float).
    pub flow_v: Texture,
    _ix: Texture,
    _iy: Texture,
    _it: Texture,
    _u_other: Texture,
    _v_other: Texture,
}

impl DenseFlow {
    pub fn new(ctx: &Context) -> Result<Self> {
        let deriv_name = objc2_foundation::ns_string!("flow_derivatives");
        let iter_name = objc2_foundation::ns_string!("horn_schunck_iterate");

        let deriv_func = ctx
            .library()
            .newFunctionWithName(deriv_name)
            .ok_or_else(|| Error::ShaderMissing("flow_derivatives".into()))?;
        let iter_func = ctx
            .library()
            .newFunctionWithName(iter_name)
            .ok_or_else(|| Error::ShaderMissing("horn_schunck_iterate".into()))?;

        let deriv_pipeline = ctx
            .device()
            .newComputePipelineStateWithFunction_error(&deriv_func)
            .map_err(|e| Error::PipelineCompile(format!("flow_derivatives: {e}")))?;
        let iter_pipeline = ctx
            .device()
            .newComputePipelineStateWithFunction_error(&iter_func)
            .map_err(|e| Error::PipelineCompile(format!("horn_schunck_iterate: {e}")))?;

        Ok(Self {
            deriv_pipeline,
            iter_pipeline,
        })
    }

    /// Computes per-pixel optical flow between two grayscale frames.
    pub fn compute(
        &self,
        ctx: &Context,
        frame0: &Texture,
        frame1: &Texture,
        config: &DenseFlowConfig,
    ) -> Result<DenseFlowResult> {
        let w = frame0.width();
        let h = frame0.height();

        let params = FlowParams {
            width: w,
            height: h,
            alpha: config.alpha,
        };

        let ix = Texture::intermediate_r32float(ctx.device(), w, h)?;
        let iy = Texture::intermediate_r32float(ctx.device(), w, h)?;
        let it = Texture::intermediate_r32float(ctx.device(), w, h)?;

        let u_a = Texture::intermediate_r32float(ctx.device(), w, h)?;
        let v_a = Texture::intermediate_r32float(ctx.device(), w, h)?;
        let u_b = Texture::intermediate_r32float(ctx.device(), w, h)?;
        let v_b = Texture::intermediate_r32float(ctx.device(), w, h)?;

        let cmd_buf = ctx
            .queue()
            .commandBuffer()
            .ok_or(Error::Gpu("failed to create command buffer".into()))?;

        Self::encode_passes(
            &self.deriv_pipeline,
            &self.iter_pipeline,
            &cmd_buf,
            frame0,
            frame1,
            &ix,
            &iy,
            &it,
            &u_a,
            &v_a,
            &u_b,
            &v_b,
            &params,
            config.iterations,
            w,
            h,
        )?;

        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        if config.iterations.is_multiple_of(2) {
            Ok(DenseFlowResult {
                flow_u: u_a,
                flow_v: v_a,
            })
        } else {
            Ok(DenseFlowResult {
                flow_u: u_b,
                flow_v: v_b,
            })
        }
    }

    /// Encodes the full flow computation without committing.
    pub fn encode(
        &self,
        ctx: &Context,
        cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
        frame0: &Texture,
        frame1: &Texture,
        config: &DenseFlowConfig,
    ) -> Result<DenseFlowEncodedState> {
        let w = frame0.width();
        let h = frame0.height();

        let params = FlowParams {
            width: w,
            height: h,
            alpha: config.alpha,
        };

        let ix = Texture::intermediate_r32float(ctx.device(), w, h)?;
        let iy = Texture::intermediate_r32float(ctx.device(), w, h)?;
        let it = Texture::intermediate_r32float(ctx.device(), w, h)?;

        let u_a = Texture::intermediate_r32float(ctx.device(), w, h)?;
        let v_a = Texture::intermediate_r32float(ctx.device(), w, h)?;
        let u_b = Texture::intermediate_r32float(ctx.device(), w, h)?;
        let v_b = Texture::intermediate_r32float(ctx.device(), w, h)?;

        Self::encode_passes(
            &self.deriv_pipeline,
            &self.iter_pipeline,
            cmd_buf,
            frame0,
            frame1,
            &ix,
            &iy,
            &it,
            &u_a,
            &v_a,
            &u_b,
            &v_b,
            &params,
            config.iterations,
            w,
            h,
        )?;

        if config.iterations.is_multiple_of(2) {
            Ok(DenseFlowEncodedState {
                flow_u: u_a,
                flow_v: v_a,
                _ix: ix,
                _iy: iy,
                _it: it,
                _u_other: u_b,
                _v_other: v_b,
            })
        } else {
            Ok(DenseFlowEncodedState {
                flow_u: u_b,
                flow_v: v_b,
                _ix: ix,
                _iy: iy,
                _it: it,
                _u_other: u_a,
                _v_other: v_a,
            })
        }
    }

    /// Encodes derivative computation and Jacobi iterations.
    #[allow(clippy::too_many_arguments)]
    fn encode_passes(
        deriv_pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        iter_pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
        frame0: &Texture,
        frame1: &Texture,
        ix: &Texture,
        iy: &Texture,
        it: &Texture,
        u_a: &Texture,
        v_a: &Texture,
        u_b: &Texture,
        v_b: &Texture,
        params: &FlowParams,
        iterations: u32,
        w: u32,
        h: u32,
    ) -> Result<()> {
        {
            let encoder = cmd_buf
                .computeCommandEncoder()
                .ok_or(Error::Gpu("failed to create compute encoder".into()))?;
            unsafe {
                encoder.setComputePipelineState(deriv_pipeline);
                encoder.setTexture_atIndex(Some(frame0.raw()), 0);
                encoder.setTexture_atIndex(Some(frame1.raw()), 1);
                encoder.setTexture_atIndex(Some(ix.raw()), 2);
                encoder.setTexture_atIndex(Some(iy.raw()), 3);
                encoder.setTexture_atIndex(Some(it.raw()), 4);
                encoder.setBytes_length_atIndex(
                    NonNull::new_unchecked(params as *const FlowParams as *mut c_void),
                    mem::size_of::<FlowParams>(),
                    0,
                );
                let (grid, tg) = Self::grid_2d(deriv_pipeline, w, h);
                encoder.dispatchThreads_threadsPerThreadgroup(grid, tg);
            }
            encoder.endEncoding();
        }

        for i in 0..iterations {
            let (u_in, v_in, u_out, v_out) = if i % 2 == 0 {
                (u_a, v_a, u_b, v_b)
            } else {
                (u_b, v_b, u_a, v_a)
            };

            let encoder = cmd_buf
                .computeCommandEncoder()
                .ok_or(Error::Gpu("failed to create compute encoder".into()))?;
            unsafe {
                encoder.setComputePipelineState(iter_pipeline);
                encoder.setTexture_atIndex(Some(ix.raw()), 0);
                encoder.setTexture_atIndex(Some(iy.raw()), 1);
                encoder.setTexture_atIndex(Some(it.raw()), 2);
                encoder.setTexture_atIndex(Some(u_in.raw()), 3);
                encoder.setTexture_atIndex(Some(v_in.raw()), 4);
                encoder.setTexture_atIndex(Some(u_out.raw()), 5);
                encoder.setTexture_atIndex(Some(v_out.raw()), 6);
                encoder.setBytes_length_atIndex(
                    NonNull::new_unchecked(params as *const FlowParams as *mut c_void),
                    mem::size_of::<FlowParams>(),
                    0,
                );
                let (grid, tg) = Self::grid_2d(iter_pipeline, w, h);
                encoder.dispatchThreads_threadsPerThreadgroup(grid, tg);
            }
            encoder.endEncoding();
        }

        Ok(())
    }

    fn grid_2d(
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        w: u32,
        h: u32,
    ) -> (MTLSize, MTLSize) {
        let tew = pipeline.threadExecutionWidth();
        let max_tg = pipeline.maxTotalThreadsPerThreadgroup();
        let tg_h = (max_tg / tew).max(1);
        (
            MTLSize {
                width: w as usize,
                height: h as usize,
                depth: 1,
            },
            MTLSize {
                width: tew,
                height: tg_h,
                depth: 1,
            },
        )
    }
}

unsafe impl Send for DenseFlow {}
unsafe impl Sync for DenseFlow {}
