//! Connected-components labeling via iterative label propagation.
//!
//! No `encode()` method is provided because labeling requires iterative
//! GPU dispatches until convergence, with a CPU-side convergence check
//! between passes.

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
use crate::types::CCLParams;

/// Configuration for connected-components labeling.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct CCLConfig {
    /// Foreground threshold (0.0--1.0); pixels above are foreground.
    pub threshold: f32,
    /// Safety limit on propagation iterations.
    pub max_iterations: u32,
}

impl Default for CCLConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            max_iterations: 256,
        }
    }
}

/// Connected-components labeling result.
pub struct CCLResult {
    /// Row-major label buffer. `0` = background, `>0` = component ID.
    pub labels: Vec<u32>,
    /// Unique component count (excluding background).
    pub n_components: u32,
    /// Iterations until convergence.
    pub iterations: u32,
}

/// Connected-components labeling pipelines. Requires CPU readback for convergence checks.
pub struct ConnectedComponents {
    init_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    iterate_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl ConnectedComponents {
    pub fn new(ctx: &Context) -> Result<Self> {
        let init_name = objc2_foundation::ns_string!("ccl_init");
        let iter_name = objc2_foundation::ns_string!("ccl_iterate");

        let init_func = ctx
            .library()
            .newFunctionWithName(init_name)
            .ok_or(Error::ShaderMissing("ccl_init".into()))?;
        let iter_func = ctx
            .library()
            .newFunctionWithName(iter_name)
            .ok_or(Error::ShaderMissing("ccl_iterate".into()))?;

        let init_pipeline = ctx
            .device()
            .newComputePipelineStateWithFunction_error(&init_func)
            .map_err(|e| Error::PipelineCompile(format!("ccl_init: {e}")))?;
        let iterate_pipeline = ctx
            .device()
            .newComputePipelineStateWithFunction_error(&iter_func)
            .map_err(|e| Error::PipelineCompile(format!("ccl_iterate: {e}")))?;

        Ok(Self {
            init_pipeline,
            iterate_pipeline,
        })
    }

    /// Labels connected components in a grayscale image.
    pub fn label(&self, ctx: &Context, input: &Texture, config: &CCLConfig) -> Result<CCLResult> {
        let w = input.width();
        let h = input.height();
        let n_pixels = (w as usize) * (h as usize);

        let params = CCLParams {
            width: w,
            height: h,
            threshold: config.threshold,
        };

        let labels_a = vx_gpu::UnifiedBuffer::<u32>::new(ctx.device(), n_pixels)?;
        let labels_b = vx_gpu::UnifiedBuffer::<u32>::new(ctx.device(), n_pixels)?;
        let mut changed_buf = vx_gpu::UnifiedBuffer::<u32>::new(ctx.device(), 1)?;

        {
            let cmd_buf = ctx
                .queue()
                .commandBuffer()
                .ok_or(Error::Gpu("failed to create command buffer".into()))?;
            let encoder = cmd_buf
                .computeCommandEncoder()
                .ok_or(Error::Gpu("failed to create compute encoder".into()))?;

            unsafe {
                encoder.setComputePipelineState(&self.init_pipeline);
                encoder.setTexture_atIndex(Some(input.raw()), 0);
                encoder.setBuffer_offset_atIndex(Some(labels_a.metal_buffer()), 0, 0);
                encoder.setBytes_length_atIndex(
                    NonNull::new_unchecked(&params as *const CCLParams as *mut c_void),
                    mem::size_of::<CCLParams>(),
                    1,
                );

                let tew = self.init_pipeline.threadExecutionWidth();
                let max_tg = self.init_pipeline.maxTotalThreadsPerThreadgroup();
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

            encoder.endEncoding();
            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();
        }

        let mut read_a = true;
        let mut iterations = 0u32;

        for _ in 0..config.max_iterations {
            changed_buf.as_mut_slice()[0] = 0;

            let (src_buf, dst_buf) = if read_a {
                (labels_a.metal_buffer(), labels_b.metal_buffer())
            } else {
                (labels_b.metal_buffer(), labels_a.metal_buffer())
            };

            let cmd_buf = ctx
                .queue()
                .commandBuffer()
                .ok_or(Error::Gpu("failed to create command buffer".into()))?;
            let encoder = cmd_buf
                .computeCommandEncoder()
                .ok_or(Error::Gpu("failed to create compute encoder".into()))?;

            unsafe {
                encoder.setComputePipelineState(&self.iterate_pipeline);
                encoder.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(dst_buf), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(changed_buf.metal_buffer()), 0, 2);
                encoder.setBytes_length_atIndex(
                    NonNull::new_unchecked(&params as *const CCLParams as *mut c_void),
                    mem::size_of::<CCLParams>(),
                    3,
                );

                let tew = self.iterate_pipeline.threadExecutionWidth();
                let max_tg = self.iterate_pipeline.maxTotalThreadsPerThreadgroup();
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

            encoder.endEncoding();
            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();

            iterations += 1;
            read_a = !read_a;

            if changed_buf.as_slice()[0] == 0 {
                break;
            }
        }

        let final_labels = if read_a {
            labels_a.as_slice()
        } else {
            labels_b.as_slice()
        };
        let labels = final_labels.to_vec();

        let mut unique: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for &l in &labels {
            if l > 0 {
                unique.insert(l);
            }
        }
        let n_components = unique.len() as u32;

        Ok(CCLResult {
            labels,
            n_components,
            iterations,
        })
    }
}

unsafe impl Send for ConnectedComponents {}
unsafe impl Sync for ConnectedComponents {}
