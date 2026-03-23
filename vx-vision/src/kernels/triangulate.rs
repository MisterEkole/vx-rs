//! GPU-accelerated midpoint triangulation from two-view correspondences.

use core::ffi::c_void;
use core::ptr::NonNull;
use std::mem;

use bytemuck::{Pod, Zeroable};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLLibrary, MTLSize,
};

use crate::context::Context;
use crate::error::{Error, Result};
use vx_gpu::UnifiedBuffer;

use crate::types::PointXYZ;
#[cfg(feature = "reconstruction")]
use crate::types_3d::{CameraExtrinsics, CameraIntrinsics, Point3D, PointCloud};

/// A 2D-2D match for triangulation.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Match2D {
    pub u1: f32,
    pub v1: f32,
    pub u2: f32,
    pub v2: f32,
}

/// GPU parameters for triangulation.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct TriangulateParams {
    pub n_matches: u32,
    pub max_points: u32,
    pub fx1: f32,
    pub fy1: f32,
    pub cx1: f32,
    pub cy1: f32,
    pub fx2: f32,
    pub fy2: f32,
    pub cx2: f32,
    pub cy2: f32,
    pub _pad: [f32; 2], // align float4 pose rows to 16-byte boundary
    pub pose1_row0: [f32; 4],
    pub pose1_row1: [f32; 4],
    pub pose1_row2: [f32; 4],
    pub pose2_row0: [f32; 4],
    pub pose2_row1: [f32; 4],
    pub pose2_row2: [f32; 4],
}

/// Compiled triangulation pipeline.
pub struct Triangulator {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

unsafe impl Send for Triangulator {}
unsafe impl Sync for Triangulator {}

impl Triangulator {
    pub fn new(ctx: &Context) -> Result<Self> {
        let func = ctx
            .library()
            .newFunctionWithName(objc2_foundation::ns_string!("triangulate_midpoint"))
            .ok_or(Error::ShaderMissing("triangulate_midpoint".into()))?;
        let pipeline = ctx
            .device()
            .newComputePipelineStateWithFunction_error(&func)
            .map_err(|e| Error::PipelineCompile(format!("triangulate_midpoint: {e}")))?;
        Ok(Self { pipeline })
    }

    /// Triangulates 3D points from 2D-2D correspondences. Synchronous.
    #[cfg(feature = "reconstruction")]
    pub fn triangulate(
        &self,
        ctx: &Context,
        matches: &[Match2D],
        intrinsics1: &CameraIntrinsics,
        intrinsics2: &CameraIntrinsics,
        pose1: &CameraExtrinsics,
        pose2: &CameraExtrinsics,
    ) -> Result<PointCloud> {
        let n = matches.len();
        if n == 0 {
            return Ok(PointCloud::new());
        }

        let mut match_buf: UnifiedBuffer<Match2D> = UnifiedBuffer::new(ctx.device(), n)?;
        match_buf.write(matches);

        let point_buf: UnifiedBuffer<PointXYZ> = UnifiedBuffer::new(ctx.device(), n)?;
        let mut count_buf: UnifiedBuffer<u32> = UnifiedBuffer::new(ctx.device(), 1)?;
        count_buf.write(&[0u32]);

        let rows1 = pose1.to_gpu_rows();
        let rows2 = pose2.to_gpu_rows();

        let params = TriangulateParams {
            n_matches: n as u32,
            max_points: n as u32,
            fx1: intrinsics1.fx,
            fy1: intrinsics1.fy,
            cx1: intrinsics1.cx,
            cy1: intrinsics1.cy,
            fx2: intrinsics2.fx,
            fy2: intrinsics2.fy,
            cx2: intrinsics2.cx,
            cy2: intrinsics2.cy,
            _pad: [0.0; 2],
            pose1_row0: rows1[0],
            pose1_row1: rows1[1],
            pose1_row2: rows1[2],
            pose2_row0: rows2[0],
            pose2_row1: rows2[1],
            pose2_row2: rows2[2],
        };

        let _mg = match_buf.gpu_guard();
        let _pg = point_buf.gpu_guard();
        let _cg = count_buf.gpu_guard();

        let cmd_buf = ctx
            .queue()
            .commandBuffer()
            .ok_or(Error::Gpu("failed to create command buffer".into()))?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or(Error::Gpu("failed to create compute encoder".into()))?;

        unsafe {
            encoder.setComputePipelineState(&self.pipeline);
            encoder.setBuffer_offset_atIndex(Some(match_buf.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(point_buf.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(count_buf.metal_buffer()), 0, 2);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(&params as *const TriangulateParams as *mut c_void),
                mem::size_of::<TriangulateParams>(),
                3,
            );

            let tew = self.pipeline.threadExecutionWidth();
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

        drop((_mg, _pg, _cg));

        let n_points = (count_buf.as_slice()[0] as usize).min(n);
        let pts = &point_buf.as_slice()[..n_points];
        let cloud_points: Vec<Point3D> = pts
            .iter()
            .map(|p| Point3D {
                position: [p.x, p.y, p.z],
                color: [200, 200, 200, 255],
                normal: [0.0; 3],
            })
            .collect();

        Ok(PointCloud {
            points: cloud_points,
        })
    }
}
