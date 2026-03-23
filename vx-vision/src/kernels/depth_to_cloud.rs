//! GPU-accelerated depth map to point cloud unprojection.

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
use crate::types::{DepthToCloudParams, GpuPoint3D};
use vx_gpu::UnifiedBuffer;

#[cfg(feature = "reconstruction")]
use crate::types_3d::{DepthMap, PointCloud};

/// Configuration for depth-to-point-cloud unprojection.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct DepthToCloudConfig {
    /// Minimum valid depth value.
    pub min_depth: f32,
    /// Maximum valid depth value.
    pub max_depth: f32,
    /// Scale factor applied to raw depth values.
    pub depth_scale: f32,
    /// Maximum number of output points.
    pub max_points: u32,
}

impl Default for DepthToCloudConfig {
    fn default() -> Self {
        Self {
            min_depth: 0.1,
            max_depth: 10.0,
            depth_scale: 1.0,
            max_points: 1_000_000,
        }
    }
}

impl DepthToCloudConfig {
    pub fn new(min_depth: f32, max_depth: f32) -> Self {
        Self {
            min_depth,
            max_depth,
            ..Default::default()
        }
    }
}

/// Compiled depth-to-cloud pipeline.
pub struct DepthToCloud {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

unsafe impl Send for DepthToCloud {}
unsafe impl Sync for DepthToCloud {}

impl DepthToCloud {
    pub fn new(ctx: &Context) -> Result<Self> {
        let func = ctx
            .library()
            .newFunctionWithName(objc2_foundation::ns_string!("depth_to_cloud"))
            .ok_or(Error::ShaderMissing("depth_to_cloud".into()))?;
        let pipeline = ctx
            .device()
            .newComputePipelineStateWithFunction_error(&func)
            .map_err(|e| Error::PipelineCompile(format!("depth_to_cloud: {e}")))?;
        Ok(Self { pipeline })
    }

    /// Unprojects a depth map to a point cloud. Synchronous.
    #[cfg(feature = "reconstruction")]
    pub fn compute(
        &self,
        ctx: &Context,
        depth: &DepthMap,
        color: Option<&Texture>,
        config: &DepthToCloudConfig,
    ) -> Result<PointCloud> {
        let intrinsics = depth.intrinsics();
        self.compute_raw(
            ctx,
            depth.texture(),
            color,
            intrinsics.fx,
            intrinsics.fy,
            intrinsics.cx,
            intrinsics.cy,
            config,
        )
    }

    /// Unprojects a depth texture to a point cloud with explicit intrinsics. Synchronous.
    #[cfg(feature = "reconstruction")]
    pub fn compute_raw(
        &self,
        ctx: &Context,
        depth_tex: &Texture,
        color_tex: Option<&Texture>,
        fx: f32,
        fy: f32,
        cx: f32,
        cy: f32,
        config: &DepthToCloudConfig,
    ) -> Result<PointCloud> {
        let w = depth_tex.width();
        let h = depth_tex.height();
        let max_pts = config.max_points.min(w * h);

        let point_buf: UnifiedBuffer<GpuPoint3D> =
            UnifiedBuffer::new(ctx.device(), max_pts as usize)?;
        let mut count_buf: UnifiedBuffer<u32> = UnifiedBuffer::new(ctx.device(), 1)?;
        count_buf.write(&[0u32]);

        let params = DepthToCloudParams {
            fx,
            fy,
            cx,
            cy,
            min_depth: config.min_depth,
            max_depth: config.max_depth,
            depth_scale: config.depth_scale,
            width: w,
            height: h,
            max_points: max_pts,
        };

        let _pg = point_buf.gpu_guard();
        let _cg = count_buf.gpu_guard();

        let cmd_buf = ctx
            .queue()
            .commandBuffer()
            .ok_or(Error::Gpu("failed to create command buffer".into()))?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or(Error::Gpu("failed to create compute encoder".into()))?;

        // Create a 1x1 dummy texture if no color texture provided
        let dummy_color;
        let color_ref = match color_tex {
            Some(tex) => tex,
            None => {
                dummy_color = Texture::output_gray8(ctx.device(), 1, 1)?;
                &dummy_color
            }
        };

        Self::encode_into(
            &self.pipeline,
            &encoder,
            depth_tex,
            color_ref,
            &point_buf,
            &count_buf,
            &params,
            w,
            h,
        );

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        drop((_pg, _cg));

        let n_points = (count_buf.as_slice()[0] as usize).min(max_pts as usize);
        let gpu_points = &point_buf.as_slice()[..n_points];
        Ok(PointCloud::from_gpu_points(gpu_points))
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_into(
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        depth_tex: &Texture,
        color_tex: &Texture,
        point_buf: &UnifiedBuffer<GpuPoint3D>,
        count_buf: &UnifiedBuffer<u32>,
        params: &DepthToCloudParams,
        w: u32,
        h: u32,
    ) {
        unsafe {
            encoder.setComputePipelineState(pipeline);
            encoder.setTexture_atIndex(Some(depth_tex.raw()), 0);
            encoder.setTexture_atIndex(Some(color_tex.raw()), 1);
            encoder.setBuffer_offset_atIndex(Some(point_buf.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(count_buf.metal_buffer()), 0, 1);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(params as *const DepthToCloudParams as *mut c_void),
                mem::size_of::<DepthToCloudParams>(),
                2,
            );

            let tew = pipeline.threadExecutionWidth();
            let max_tg = pipeline.maxTotalThreadsPerThreadgroup();
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
    }
}
