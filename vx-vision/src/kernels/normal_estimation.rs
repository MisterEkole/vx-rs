//! Normal estimation for organized (depth map) and unorganized point clouds.

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
use crate::types::{NormalEstParams, NormalEstUnorgParams, PointXYZ};
use vx_gpu::UnifiedBuffer;

#[cfg(feature = "reconstruction")]
use crate::types_3d::PointCloud;

/// Configuration for normal estimation.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct NormalEstimatorConfig {
    /// Search radius for unorganized mode.
    pub radius: f32,
    /// Maximum neighbors to consider.
    pub max_neighbors: u32,
}

impl Default for NormalEstimatorConfig {
    fn default() -> Self {
        Self {
            radius: 0.1,
            max_neighbors: 30,
        }
    }
}

/// Compiled normal estimation pipelines.
pub struct NormalEstimator {
    organized_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    unorganized_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

unsafe impl Send for NormalEstimator {}
unsafe impl Sync for NormalEstimator {}

impl NormalEstimator {
    pub fn new(ctx: &Context) -> Result<Self> {
        let org_func = ctx
            .library()
            .newFunctionWithName(objc2_foundation::ns_string!("normal_estimate_organized"))
            .ok_or(Error::ShaderMissing("normal_estimate_organized".into()))?;
        let unorg_func = ctx
            .library()
            .newFunctionWithName(objc2_foundation::ns_string!("normal_estimate_unorganized"))
            .ok_or(Error::ShaderMissing("normal_estimate_unorganized".into()))?;

        let organized_pipeline = ctx
            .device()
            .newComputePipelineStateWithFunction_error(&org_func)
            .map_err(|e| Error::PipelineCompile(format!("normal_estimate_organized: {e}")))?;
        let unorganized_pipeline = ctx
            .device()
            .newComputePipelineStateWithFunction_error(&unorg_func)
            .map_err(|e| Error::PipelineCompile(format!("normal_estimate_unorganized: {e}")))?;

        Ok(Self {
            organized_pipeline,
            unorganized_pipeline,
        })
    }

    /// Estimates normals from a depth map (organized path). Synchronous.
    pub fn compute_from_depth(
        &self,
        ctx: &Context,
        depth: &Texture,
        output: &Texture,
        fx: f32,
        fy: f32,
        cx: f32,
        cy: f32,
    ) -> Result<()> {
        let w = depth.width();
        let h = depth.height();
        let params = NormalEstParams {
            fx,
            fy,
            cx,
            cy,
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

        unsafe {
            encoder.setComputePipelineState(&self.organized_pipeline);
            encoder.setTexture_atIndex(Some(depth.raw()), 0);
            encoder.setTexture_atIndex(Some(output.raw()), 1);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(&params as *const NormalEstParams as *mut c_void),
                mem::size_of::<NormalEstParams>(),
                0,
            );

            let tew = self.organized_pipeline.threadExecutionWidth();
            let max_tg = self.organized_pipeline.maxTotalThreadsPerThreadgroup();
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
        Ok(())
    }

    /// Estimates normals for an unorganized point cloud. Synchronous.
    #[cfg(feature = "reconstruction")]
    pub fn compute(
        &self,
        ctx: &Context,
        cloud: &PointCloud,
        config: &NormalEstimatorConfig,
    ) -> Result<Vec<[f32; 3]>> {
        let n = cloud.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        // Convert PointCloud to PointXYZ buffer
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

        let normal_buf: UnifiedBuffer<f32> = UnifiedBuffer::new(ctx.device(), n * 3)?;

        let params = NormalEstUnorgParams {
            n_points: n as u32,
            radius: config.radius,
            max_neighbors: config.max_neighbors,
        };

        let _pg = point_buf.gpu_guard();
        let _ng = normal_buf.gpu_guard();

        let cmd_buf = ctx
            .queue()
            .commandBuffer()
            .ok_or(Error::Gpu("failed to create command buffer".into()))?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or(Error::Gpu("failed to create compute encoder".into()))?;

        unsafe {
            encoder.setComputePipelineState(&self.unorganized_pipeline);
            encoder.setBuffer_offset_atIndex(Some(point_buf.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(normal_buf.metal_buffer()), 0, 1);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(&params as *const NormalEstUnorgParams as *mut c_void),
                mem::size_of::<NormalEstUnorgParams>(),
                2,
            );

            let tew = self.unorganized_pipeline.threadExecutionWidth();
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

        drop((_pg, _ng));

        let raw = normal_buf.as_slice();
        let normals: Vec<[f32; 3]> = (0..n)
            .map(|i| [raw[i * 3], raw[i * 3 + 1], raw[i * 3 + 2]])
            .collect();
        Ok(normals)
    }
}
