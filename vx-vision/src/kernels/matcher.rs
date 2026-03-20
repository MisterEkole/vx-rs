//! Brute-force descriptor matching with Hamming distance.

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
use crate::types::{MatcherParams, MatchResult};

/// Configuration for brute-force matching.
#[derive(Clone, Debug)]
pub struct MatchConfig {
    /// Maximum accepted Hamming distance (0--256 for ORB).
    pub max_hamming: u32,
    /// Lowe's ratio test threshold. Lower is stricter.
    pub ratio_thresh: f32,
}

impl Default for MatchConfig {
    fn default() -> Self {
        Self {
            max_hamming: 64,
            ratio_thresh: 0.75,
        }
    }
}

/// Brute-force matcher compute pipelines.
pub struct BruteMatcher {
    hamming_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    extract_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl BruteMatcher {
    pub fn new(ctx: &Context) -> Result<Self, String> {
        let ham_name = objc2_foundation::ns_string!("brutematch_hamming");
        let ext_name = objc2_foundation::ns_string!("brutematch_extract");

        let ham_func = ctx.library().newFunctionWithName(ham_name)
            .ok_or_else(|| "Missing kernel function 'brutematch_hamming'".to_string())?;
        let ext_func = ctx.library().newFunctionWithName(ext_name)
            .ok_or_else(|| "Missing kernel function 'brutematch_extract'".to_string())?;

        let hamming_pipeline = ctx.device()
            .newComputePipelineStateWithFunction_error(&ham_func)
            .map_err(|e| format!("Failed to create brutematch_hamming pipeline: {e}"))?;
        let extract_pipeline = ctx.device()
            .newComputePipelineStateWithFunction_error(&ext_func)
            .map_err(|e| format!("Failed to create brutematch_extract pipeline: {e}"))?;

        Ok(Self { hamming_pipeline, extract_pipeline })
    }

    /// Matches two sets of ORB descriptors (256-bit = 8 x u32 each).
    ///
    /// `query_desc` and `train_desc` are flat `u32` arrays where every 8
    /// consecutive values form one descriptor. Returns pairs passing both
    /// the distance threshold and ratio test.
    pub fn match_descriptors(
        &self,
        ctx:        &Context,
        query_desc: &[u32],
        train_desc: &[u32],
        config:     &MatchConfig,
    ) -> Result<Vec<MatchResult>, String> {
        let n_query = (query_desc.len() / 8) as u32;
        let n_train = (train_desc.len() / 8) as u32;

        if n_query == 0 || n_train == 0 {
            return Ok(Vec::new());
        }

        let params = MatcherParams {
            n_query,
            n_train,
            max_hamming:  config.max_hamming,
            ratio_thresh: config.ratio_thresh,
        };

        let mut query_buf = vx_core::UnifiedBuffer::<u32>::new(ctx.device(), query_desc.len())?;
        query_buf.as_mut_slice().copy_from_slice(query_desc);

        let mut train_buf = vx_core::UnifiedBuffer::<u32>::new(ctx.device(), train_desc.len())?;
        train_buf.as_mut_slice().copy_from_slice(train_desc);

        // Distance matrix: n_query x n_train, u16
        let dist_size = (n_query as usize) * (n_train as usize);
        let dist_buf = vx_core::UnifiedBuffer::<u16>::new(ctx.device(), dist_size)?;

        let match_buf = vx_core::UnifiedBuffer::<MatchResult>::new(ctx.device(), n_query as usize)?;
        let mut count_buf = vx_core::UnifiedBuffer::<u32>::new(ctx.device(), 1)?;
        count_buf.as_mut_slice()[0] = 0;

        // Pass 1: Hamming distance matrix (2D dispatch over query x train)
        {
            let cmd_buf = ctx.queue().commandBuffer()
                .ok_or("Failed to create command buffer")?;
            let encoder = cmd_buf.computeCommandEncoder()
                .ok_or("Failed to create compute encoder")?;

            unsafe {
                encoder.setComputePipelineState(&self.hamming_pipeline);
                encoder.setBuffer_offset_atIndex(Some(query_buf.metal_buffer()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(train_buf.metal_buffer()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(dist_buf.metal_buffer()),  0, 2);
                encoder.setBytes_length_atIndex(
                    NonNull::new_unchecked(&params as *const MatcherParams as *mut c_void),
                    mem::size_of::<MatcherParams>(),
                    3,
                );

                let tew    = self.hamming_pipeline.threadExecutionWidth();
                let max_tg = self.hamming_pipeline.maxTotalThreadsPerThreadgroup();
                let tg_h   = (max_tg / tew).max(1);
                let grid    = MTLSize { width: n_query as usize, height: n_train as usize, depth: 1 };
                let tg_size = MTLSize { width: tew,              height: tg_h,             depth: 1 };
                encoder.dispatchThreads_threadsPerThreadgroup(grid, tg_size);
            }

            encoder.endEncoding();
            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();
        }

        // Pass 2: extract best match per query with ratio test
        {
            let cmd_buf = ctx.queue().commandBuffer()
                .ok_or("Failed to create command buffer")?;
            let encoder = cmd_buf.computeCommandEncoder()
                .ok_or("Failed to create compute encoder")?;

            unsafe {
                encoder.setComputePipelineState(&self.extract_pipeline);
                encoder.setBuffer_offset_atIndex(Some(dist_buf.metal_buffer()),  0, 0);
                encoder.setBuffer_offset_atIndex(Some(match_buf.metal_buffer()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(count_buf.metal_buffer()), 0, 2);
                encoder.setBytes_length_atIndex(
                    NonNull::new_unchecked(&params as *const MatcherParams as *mut c_void),
                    mem::size_of::<MatcherParams>(),
                    3,
                );

                let tew    = self.extract_pipeline.threadExecutionWidth();
                let max_tg = self.extract_pipeline.maxTotalThreadsPerThreadgroup();
                let grid    = MTLSize { width: n_query as usize, height: 1, depth: 1 };
                let tg_size = MTLSize { width: tew.min(max_tg),  height: 1, depth: 1 };
                encoder.dispatchThreads_threadsPerThreadgroup(grid, tg_size);
            }

            encoder.endEncoding();
            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();
        }

        let n_matches = count_buf.as_slice()[0] as usize;
        let n_matches = n_matches.min(n_query as usize);
        Ok(match_buf.as_slice()[..n_matches].to_vec())
    }
}
