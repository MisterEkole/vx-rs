//! Voxel grid downsampling for point clouds.

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
use crate::types::{PointXYZ, VoxelDownsampleParams};
use vx_gpu::UnifiedBuffer;

#[cfg(feature = "reconstruction")]
use crate::types_3d::{Point3D, PointCloud};

/// Configuration for voxel downsampling.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct VoxelDownsampleConfig {
    /// Size of each voxel cube in world units.
    pub voxel_size: f32,
}

impl Default for VoxelDownsampleConfig {
    fn default() -> Self {
        Self { voxel_size: 0.05 }
    }
}

impl VoxelDownsampleConfig {
    pub fn new(voxel_size: f32) -> Self {
        Self { voxel_size }
    }
}

/// Compiled voxel downsample pipelines.
pub struct VoxelDownsample {
    hash_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    average_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

unsafe impl Send for VoxelDownsample {}
unsafe impl Sync for VoxelDownsample {}

impl VoxelDownsample {
    pub fn new(ctx: &Context) -> Result<Self> {
        let hash_func = ctx
            .library()
            .newFunctionWithName(objc2_foundation::ns_string!("voxel_hash_assign"))
            .ok_or(Error::ShaderMissing("voxel_hash_assign".into()))?;
        let avg_func = ctx
            .library()
            .newFunctionWithName(objc2_foundation::ns_string!("voxel_average"))
            .ok_or(Error::ShaderMissing("voxel_average".into()))?;

        let hash_pipeline = ctx
            .device()
            .newComputePipelineStateWithFunction_error(&hash_func)
            .map_err(|e| Error::PipelineCompile(format!("voxel_hash_assign: {e}")))?;
        let average_pipeline = ctx
            .device()
            .newComputePipelineStateWithFunction_error(&avg_func)
            .map_err(|e| Error::PipelineCompile(format!("voxel_average: {e}")))?;

        Ok(Self {
            hash_pipeline,
            average_pipeline,
        })
    }

    /// Downsamples a point cloud by averaging points within each voxel. Synchronous.
    #[cfg(feature = "reconstruction")]
    pub fn downsample(
        &self,
        ctx: &Context,
        cloud: &PointCloud,
        config: &VoxelDownsampleConfig,
    ) -> Result<PointCloud> {
        let n = cloud.len();
        if n == 0 {
            return Ok(PointCloud::new());
        }

        // Compute bounds for origin
        let (min_bound, _) = cloud.bounds().unwrap();

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

        // Hash table size: next power of 2 at least 2x the number of points
        let table_size = (n * 2).next_power_of_two().max(256);

        let mut point_buf: UnifiedBuffer<PointXYZ> = UnifiedBuffer::new(ctx.device(), n)?;
        point_buf.write(&points);

        let mut cell_counts: UnifiedBuffer<u32> = UnifiedBuffer::new(ctx.device(), table_size)?;
        let mut cell_sum_x: UnifiedBuffer<u32> = UnifiedBuffer::new(ctx.device(), table_size)?;
        let mut cell_sum_y: UnifiedBuffer<u32> = UnifiedBuffer::new(ctx.device(), table_size)?;
        let mut cell_sum_z: UnifiedBuffer<u32> = UnifiedBuffer::new(ctx.device(), table_size)?;

        // Zero all hash table buffers
        cell_counts.write(&vec![0u32; table_size]);
        cell_sum_x.write(&vec![0u32; table_size]);
        cell_sum_y.write(&vec![0u32; table_size]);
        cell_sum_z.write(&vec![0u32; table_size]);

        let output_buf: UnifiedBuffer<PointXYZ> = UnifiedBuffer::new(ctx.device(), n)?;
        let mut out_count: UnifiedBuffer<u32> = UnifiedBuffer::new(ctx.device(), 1)?;
        out_count.write(&[0u32]);

        let params = VoxelDownsampleParams {
            n_points: n as u32,
            voxel_size: config.voxel_size,
            origin_x: min_bound[0],
            origin_y: min_bound[1],
            origin_z: min_bound[2],
            table_size: table_size as u32,
        };

        {
            let _pg = point_buf.gpu_guard();
            let _cg = cell_counts.gpu_guard();
            let _sx = cell_sum_x.gpu_guard();
            let _sy = cell_sum_y.gpu_guard();
            let _sz = cell_sum_z.gpu_guard();

            let cmd_buf = ctx
                .queue()
                .commandBuffer()
                .ok_or(Error::Gpu("failed to create command buffer".into()))?;
            let encoder = cmd_buf
                .computeCommandEncoder()
                .ok_or(Error::Gpu("failed to create compute encoder".into()))?;

            unsafe {
                encoder.setComputePipelineState(&self.hash_pipeline);
                encoder.setBuffer_offset_atIndex(Some(point_buf.metal_buffer()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(cell_counts.metal_buffer()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(cell_sum_x.metal_buffer()), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(cell_sum_y.metal_buffer()), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(cell_sum_z.metal_buffer()), 0, 4);
                encoder.setBytes_length_atIndex(
                    NonNull::new_unchecked(&params as *const VoxelDownsampleParams as *mut c_void),
                    mem::size_of::<VoxelDownsampleParams>(),
                    5,
                );

                let tew = self.hash_pipeline.threadExecutionWidth();
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

        {
            let _cg = cell_counts.gpu_guard();
            let _sx = cell_sum_x.gpu_guard();
            let _sy = cell_sum_y.gpu_guard();
            let _sz = cell_sum_z.gpu_guard();
            let _og = output_buf.gpu_guard();
            let _oc = out_count.gpu_guard();

            let cmd_buf = ctx
                .queue()
                .commandBuffer()
                .ok_or(Error::Gpu("failed to create command buffer".into()))?;
            let encoder = cmd_buf
                .computeCommandEncoder()
                .ok_or(Error::Gpu("failed to create compute encoder".into()))?;

            unsafe {
                encoder.setComputePipelineState(&self.average_pipeline);
                encoder.setBuffer_offset_atIndex(Some(cell_counts.metal_buffer()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(cell_sum_x.metal_buffer()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(cell_sum_y.metal_buffer()), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(cell_sum_z.metal_buffer()), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(output_buf.metal_buffer()), 0, 4);
                encoder.setBuffer_offset_atIndex(Some(out_count.metal_buffer()), 0, 5);
                encoder.setBytes_length_atIndex(
                    NonNull::new_unchecked(&params as *const VoxelDownsampleParams as *mut c_void),
                    mem::size_of::<VoxelDownsampleParams>(),
                    6,
                );

                let tew = self.average_pipeline.threadExecutionWidth();
                let grid = MTLSize {
                    width: table_size,
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

        let n_out = (out_count.as_slice()[0] as usize).min(n);
        let raw_pts = &output_buf.as_slice()[..n_out];

        let result_points: Vec<Point3D> = raw_pts
            .iter()
            .map(|p| Point3D {
                position: [p.x, p.y, p.z],
                color: [255, 255, 255, 255],
                normal: [0.0; 3],
            })
            .collect();

        Ok(PointCloud {
            points: result_points,
        })
    }
}
