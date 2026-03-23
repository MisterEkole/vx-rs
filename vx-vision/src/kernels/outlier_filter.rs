//! Statistical outlier removal for point clouds.

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
use crate::types::{OutlierParams, PointXYZ};
use vx_gpu::UnifiedBuffer;

#[cfg(feature = "reconstruction")]
use crate::types_3d::PointCloud;

/// Configuration for statistical outlier removal.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct OutlierFilterConfig {
    /// Number of nearest neighbors for distance computation.
    pub k_neighbors: u32,
    /// Standard deviation multiplier. Points beyond `mean + std_ratio * std` are outliers.
    pub std_ratio: f32,
}

impl Default for OutlierFilterConfig {
    fn default() -> Self {
        Self {
            k_neighbors: 10,
            std_ratio: 2.0,
        }
    }
}

/// Compiled outlier filter pipelines.
pub struct OutlierFilter {
    distance_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    classify_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

unsafe impl Send for OutlierFilter {}
unsafe impl Sync for OutlierFilter {}

impl OutlierFilter {
    pub fn new(ctx: &Context) -> Result<Self> {
        let dist_func = ctx
            .library()
            .newFunctionWithName(objc2_foundation::ns_string!("outlier_compute_distances"))
            .ok_or(Error::ShaderMissing("outlier_compute_distances".into()))?;
        let classify_func = ctx
            .library()
            .newFunctionWithName(objc2_foundation::ns_string!("outlier_classify"))
            .ok_or(Error::ShaderMissing("outlier_classify".into()))?;

        let distance_pipeline = ctx
            .device()
            .newComputePipelineStateWithFunction_error(&dist_func)
            .map_err(|e| Error::PipelineCompile(format!("outlier_compute_distances: {e}")))?;
        let classify_pipeline = ctx
            .device()
            .newComputePipelineStateWithFunction_error(&classify_func)
            .map_err(|e| Error::PipelineCompile(format!("outlier_classify: {e}")))?;

        Ok(Self {
            distance_pipeline,
            classify_pipeline,
        })
    }

    /// Filters outliers from a point cloud. Synchronous.
    #[cfg(feature = "reconstruction")]
    pub fn filter(
        &self,
        ctx: &Context,
        cloud: &PointCloud,
        config: &OutlierFilterConfig,
    ) -> Result<PointCloud> {
        let n = cloud.len();
        if n == 0 {
            return Ok(PointCloud::new());
        }

        // Convert to GPU format
        let points: Vec<PointXYZ> = cloud
            .points
            .iter()
            .map(|p| PointXYZ {
                x: p.position[0],
                y: p.position[1],
                z: p.position[2],
                _pad: 0.0,
            })
            .collect();

        let mut point_buf: UnifiedBuffer<PointXYZ> = UnifiedBuffer::new(ctx.device(), n)?;
        point_buf.write(&points);

        let mean_dist_buf: UnifiedBuffer<f32> = UnifiedBuffer::new(ctx.device(), n)?;
        let inlier_mask_buf: UnifiedBuffer<u32> = UnifiedBuffer::new(ctx.device(), n)?;
        let mut inlier_count_buf: UnifiedBuffer<u32> = UnifiedBuffer::new(ctx.device(), 1)?;
        inlier_count_buf.write(&[0u32]);

        let params = OutlierParams {
            n_points: n as u32,
            k_neighbors: config.k_neighbors,
            std_ratio: config.std_ratio,
        };

        {
            let _pg = point_buf.gpu_guard();
            let _mg = mean_dist_buf.gpu_guard();

            let cmd_buf = ctx
                .queue()
                .commandBuffer()
                .ok_or(Error::Gpu("failed to create command buffer".into()))?;
            let encoder = cmd_buf
                .computeCommandEncoder()
                .ok_or(Error::Gpu("failed to create compute encoder".into()))?;

            unsafe {
                encoder.setComputePipelineState(&self.distance_pipeline);
                encoder.setBuffer_offset_atIndex(Some(point_buf.metal_buffer()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(mean_dist_buf.metal_buffer()), 0, 1);
                encoder.setBytes_length_atIndex(
                    NonNull::new_unchecked(&params as *const OutlierParams as *mut c_void),
                    mem::size_of::<OutlierParams>(),
                    2,
                );

                let tew = self.distance_pipeline.threadExecutionWidth();
                let grid = MTLSize {
                    width: n,
                    height: 1,
                    depth: 1,
                };
                let tg_size = MTLSize {
                    width: tew,
                    height: 1,
                    depth: 1,
                };
                encoder.dispatchThreads_threadsPerThreadgroup(grid, tg_size);
            }

            encoder.endEncoding();
            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();
        }

        // threshold from mean distances
        let dists = mean_dist_buf.as_slice();
        let global_mean: f32 = dists.iter().sum::<f32>() / n as f32;
        let variance: f32 = dists.iter().map(|d| (d - global_mean).powi(2)).sum::<f32>() / n as f32;
        let stddev = variance.sqrt();
        let threshold = global_mean + config.std_ratio * stddev;

        {
            let _mg = mean_dist_buf.gpu_guard();
            let _ig = inlier_mask_buf.gpu_guard();
            let _cg = inlier_count_buf.gpu_guard();

            let mut threshold_buf: UnifiedBuffer<f32> = UnifiedBuffer::new(ctx.device(), 1)?;
            threshold_buf.write(&[threshold]);
            let _tg = threshold_buf.gpu_guard();

            let cmd_buf = ctx
                .queue()
                .commandBuffer()
                .ok_or(Error::Gpu("failed to create command buffer".into()))?;
            let encoder = cmd_buf
                .computeCommandEncoder()
                .ok_or(Error::Gpu("failed to create compute encoder".into()))?;

            unsafe {
                encoder.setComputePipelineState(&self.classify_pipeline);
                encoder.setBuffer_offset_atIndex(Some(mean_dist_buf.metal_buffer()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(inlier_mask_buf.metal_buffer()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(inlier_count_buf.metal_buffer()), 0, 2);
                encoder.setBytes_length_atIndex(
                    NonNull::new_unchecked(&params as *const OutlierParams as *mut c_void),
                    mem::size_of::<OutlierParams>(),
                    3,
                );
                encoder.setBuffer_offset_atIndex(Some(threshold_buf.metal_buffer()), 0, 4);

                let tew = self.classify_pipeline.threadExecutionWidth();
                let grid = MTLSize {
                    width: n,
                    height: 1,
                    depth: 1,
                };
                let tg_size = MTLSize {
                    width: tew,
                    height: 1,
                    depth: 1,
                };
                encoder.dispatchThreads_threadsPerThreadgroup(grid, tg_size);
            }

            encoder.endEncoding();
            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();
        }

        // Compact inliers on CPU
        let mask = inlier_mask_buf.as_slice();
        let inliers: Vec<_> = cloud
            .points
            .iter()
            .zip(mask.iter())
            .filter(|(_, &m)| m == 1)
            .map(|(p, _)| *p)
            .collect();

        Ok(PointCloud { points: inliers })
    }
}
