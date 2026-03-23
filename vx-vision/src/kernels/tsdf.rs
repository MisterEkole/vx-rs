//! TSDF volume fusion (integrate + raycast).

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
use crate::types::{TSDFIntegrateParams, TSDFRaycastParams};

#[cfg(feature = "reconstruction")]
use crate::types_3d::{
    CameraExtrinsics, CameraIntrinsics, DepthMap, Point3D, PointCloud, VoxelGrid,
};

/// Configuration for TSDF volume.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct TSDFConfig {
    /// Number of voxels in each dimension.
    pub resolution: [u32; 3],
    /// Size of each voxel in meters.
    pub voxel_size: f32,
    /// Truncation distance in meters.
    pub truncation_distance: f32,
    /// Maximum integration weight per voxel.
    pub max_weight: f32,
    /// World-space origin of the volume.
    pub origin: [f32; 3],
}

impl Default for TSDFConfig {
    fn default() -> Self {
        Self {
            resolution: [128, 128, 128],
            voxel_size: 0.01,
            truncation_distance: 0.04,
            max_weight: 100.0,
            origin: [-0.64, -0.64, 0.0],
        }
    }
}

/// TSDF volume with GPU integration and raycasting.
#[cfg(feature = "reconstruction")]
pub struct TSDFVolume {
    volume: VoxelGrid,
    integrate_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    raycast_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    config: TSDFConfig,
}

#[cfg(feature = "reconstruction")]
unsafe impl Send for TSDFVolume {}
#[cfg(feature = "reconstruction")]
unsafe impl Sync for TSDFVolume {}

#[cfg(feature = "reconstruction")]
impl TSDFVolume {
    /// Creates a new TSDF volume and compiles GPU pipelines.
    pub fn new(ctx: &Context, config: TSDFConfig) -> Result<Self> {
        let integrate_func = ctx
            .library()
            .newFunctionWithName(objc2_foundation::ns_string!("tsdf_integrate"))
            .ok_or(Error::ShaderMissing("tsdf_integrate".into()))?;
        let raycast_func = ctx
            .library()
            .newFunctionWithName(objc2_foundation::ns_string!("tsdf_raycast"))
            .ok_or(Error::ShaderMissing("tsdf_raycast".into()))?;

        let integrate_pipeline = ctx
            .device()
            .newComputePipelineStateWithFunction_error(&integrate_func)
            .map_err(|e| Error::PipelineCompile(format!("tsdf_integrate: {e}")))?;
        let raycast_pipeline = ctx
            .device()
            .newComputePipelineStateWithFunction_error(&raycast_func)
            .map_err(|e| Error::PipelineCompile(format!("tsdf_raycast: {e}")))?;

        let volume = VoxelGrid::new(
            ctx.device(),
            config.resolution,
            config.voxel_size,
            config.origin,
        )
        .map_err(Error::from)?;

        Ok(Self {
            volume,
            integrate_pipeline,
            raycast_pipeline,
            config,
        })
    }

    /// Integrates a depth frame into the volume. Synchronous.
    pub fn integrate(
        &self,
        ctx: &Context,
        depth: &DepthMap,
        pose: &CameraExtrinsics,
    ) -> Result<()> {
        let intrinsics = depth.intrinsics();
        let rows = pose.to_gpu_rows();

        let params = TSDFIntegrateParams {
            res_x: self.config.resolution[0],
            res_y: self.config.resolution[1],
            res_z: self.config.resolution[2],
            voxel_size: self.config.voxel_size,
            origin_x: self.config.origin[0],
            origin_y: self.config.origin[1],
            origin_z: self.config.origin[2],
            truncation_dist: self.config.truncation_distance,
            max_weight: self.config.max_weight,
            fx: intrinsics.fx,
            fy: intrinsics.fy,
            cx: intrinsics.cx,
            cy: intrinsics.cy,
            img_width: depth.width(),
            img_height: depth.height(),
            _pad0: 0.0,
            pose_row0: rows[0],
            pose_row1: rows[1],
            pose_row2: rows[2],
        };

        let _tg = self.volume.tsdf.gpu_guard();
        let _wg = self.volume.weights.gpu_guard();

        let cmd_buf = ctx
            .queue()
            .commandBuffer()
            .ok_or(Error::Gpu("failed to create command buffer".into()))?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or(Error::Gpu("failed to create compute encoder".into()))?;

        unsafe {
            encoder.setComputePipelineState(&self.integrate_pipeline);
            encoder.setTexture_atIndex(Some(depth.texture().raw()), 0);
            encoder.setBuffer_offset_atIndex(Some(self.volume.tsdf.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(self.volume.weights.metal_buffer()), 0, 1);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(&params as *const TSDFIntegrateParams as *mut c_void),
                mem::size_of::<TSDFIntegrateParams>(),
                2,
            );

            let tew = self.integrate_pipeline.threadExecutionWidth();
            let max_tg = self.integrate_pipeline.maxTotalThreadsPerThreadgroup();
            let tg_h = (max_tg / tew).max(1);
            let grid = MTLSize {
                width: self.config.resolution[0] as usize,
                height: self.config.resolution[1] as usize,
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

        drop((_tg, _wg));
        Ok(())
    }

    /// Raycasts the volume to produce a synthetic depth map and normal map. Synchronous.
    pub fn raycast(
        &self,
        ctx: &Context,
        pose: &CameraExtrinsics,
        intrinsics: &CameraIntrinsics,
    ) -> Result<(Texture, Texture)> {
        let w = intrinsics.width;
        let h = intrinsics.height;

        let depth_out = Texture::output_r32float(ctx.device(), w, h)?;
        let normal_out = Texture::output_rgba8(ctx.device(), w, h)?;

        let inv_pose = pose.inverse();
        let inv_rows = inv_pose.to_gpu_rows();

        let params = TSDFRaycastParams {
            res_x: self.config.resolution[0],
            res_y: self.config.resolution[1],
            res_z: self.config.resolution[2],
            voxel_size: self.config.voxel_size,
            origin_x: self.config.origin[0],
            origin_y: self.config.origin[1],
            origin_z: self.config.origin[2],
            truncation_dist: self.config.truncation_distance,
            fx: intrinsics.fx,
            fy: intrinsics.fy,
            cx: intrinsics.cx,
            cy: intrinsics.cy,
            img_width: w,
            img_height: h,
            _pad0: [0.0; 2],
            inv_pose_row0: inv_rows[0],
            inv_pose_row1: inv_rows[1],
            inv_pose_row2: inv_rows[2],
        };

        let _tg = self.volume.tsdf.gpu_guard();

        let cmd_buf = ctx
            .queue()
            .commandBuffer()
            .ok_or(Error::Gpu("failed to create command buffer".into()))?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or(Error::Gpu("failed to create compute encoder".into()))?;

        unsafe {
            encoder.setComputePipelineState(&self.raycast_pipeline);
            encoder.setBuffer_offset_atIndex(Some(self.volume.tsdf.metal_buffer()), 0, 0);
            encoder.setTexture_atIndex(Some(depth_out.raw()), 0);
            encoder.setTexture_atIndex(Some(normal_out.raw()), 1);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(&params as *const TSDFRaycastParams as *mut c_void),
                mem::size_of::<TSDFRaycastParams>(),
                1,
            );

            let tew = self.raycast_pipeline.threadExecutionWidth();
            let max_tg = self.raycast_pipeline.maxTotalThreadsPerThreadgroup();
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

        drop(_tg);
        Ok((depth_out, normal_out))
    }

    /// Extracts surface points from the volume where TSDF crosses zero. CPU-side.
    pub fn extract_cloud(&self) -> PointCloud {
        let res = self.config.resolution;
        let tsdf = self.volume.tsdf.as_slice();
        let weights = self.volume.weights.as_slice();
        let mut points = Vec::new();

        for iz in 0..res[2] - 1 {
            for iy in 0..res[1] - 1 {
                for ix in 0..res[0] - 1 {
                    let idx = self.volume.voxel_index(ix, iy, iz);
                    let w = weights[idx];
                    if w <= 0.0 {
                        continue;
                    }

                    let val = tsdf[idx];
                    // Check neighbors for zero-crossing
                    let idx_x = self.volume.voxel_index(ix + 1, iy, iz);
                    let idx_y = self.volume.voxel_index(ix, iy + 1, iz);
                    let idx_z = self.volume.voxel_index(ix, iy, iz + 1);

                    let has_crossing = (val > 0.0 && tsdf[idx_x] < 0.0)
                        || (val < 0.0 && tsdf[idx_x] > 0.0)
                        || (val > 0.0 && tsdf[idx_y] < 0.0)
                        || (val < 0.0 && tsdf[idx_y] > 0.0)
                        || (val > 0.0 && tsdf[idx_z] < 0.0)
                        || (val < 0.0 && tsdf[idx_z] > 0.0);

                    if has_crossing {
                        let pos = self.volume.voxel_to_world(ix, iy, iz);
                        points.push(Point3D {
                            position: pos,
                            color: [200, 200, 200, 255],
                            normal: [0.0; 3],
                        });
                    }
                }
            }
        }

        PointCloud { points }
    }

    /// Returns a reference to the underlying voxel grid.
    pub fn volume(&self) -> &VoxelGrid {
        &self.volume
    }

    /// Resets the volume to its initial state.
    pub fn reset(&mut self) {
        self.volume.reset();
    }
}
