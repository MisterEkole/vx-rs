//! Point cloud renderer using Metal render pipeline.

use core::ffi::c_void;
use core::ptr::NonNull;
use std::mem;

use bytemuck::{Pod, Zeroable};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLClearColor, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLDevice, MTLLibrary,
    MTLLoadAction, MTLPixelFormat, MTLPrimitiveType, MTLRenderCommandEncoder,
    MTLRenderPassDescriptor, MTLRenderPipelineDescriptor, MTLRenderPipelineState, MTLStoreAction,
};

use crate::context::Context;
use crate::error::{Error, Result};
use crate::render_context::{Camera, RenderTarget};
use crate::types::GpuPoint3D;
use vx_gpu::UnifiedBuffer;

#[cfg(feature = "reconstruction")]
use crate::types_3d::PointCloud;

/// GPU uniforms for point cloud rendering.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct PointRenderUniforms {
    mvp: [f32; 16],
    point_size: f32,
    _pad: [f32; 3],
}

/// Compiled point cloud render pipeline.
pub struct PointCloudRenderer {
    pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
}

unsafe impl Send for PointCloudRenderer {}
unsafe impl Sync for PointCloudRenderer {}

impl PointCloudRenderer {
    /// Compiles the point cloud render pipeline.
    pub fn new(ctx: &Context) -> Result<Self> {
        let vertex_func = ctx
            .library()
            .newFunctionWithName(objc2_foundation::ns_string!("point_cloud_vertex"))
            .ok_or(Error::ShaderMissing("point_cloud_vertex".into()))?;
        let fragment_func = ctx
            .library()
            .newFunctionWithName(objc2_foundation::ns_string!("point_cloud_fragment"))
            .ok_or(Error::ShaderMissing("point_cloud_fragment".into()))?;

        let desc = MTLRenderPipelineDescriptor::new();
        desc.setVertexFunction(Some(&vertex_func));
        desc.setFragmentFunction(Some(&fragment_func));

        let color_attachment = unsafe { desc.colorAttachments().objectAtIndexedSubscript(0) };
        color_attachment.setPixelFormat(MTLPixelFormat::RGBA8Unorm);

        desc.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float);

        let pipeline = ctx
            .device()
            .newRenderPipelineStateWithDescriptor_error(&desc)
            .map_err(|e| Error::PipelineCompile(format!("point_cloud_render: {e}")))?;

        Ok(Self { pipeline })
    }

    /// Renders a point cloud to the render target. Synchronous.
    #[cfg(feature = "reconstruction")]
    pub fn render(
        &self,
        ctx: &Context,
        cloud: &PointCloud,
        camera: &Camera,
        target: &RenderTarget,
        point_size: f32,
    ) -> Result<()> {
        if cloud.is_empty() {
            return Ok(());
        }

        // Convert PointCloud to GPU buffer
        let gpu_points: Vec<GpuPoint3D> = cloud
            .points
            .iter()
            .map(|p| GpuPoint3D {
                position: p.position,
                _pad0: 0.0,
                color: p.color,
                normal: p.normal,
                _pad1: 0.0,
            })
            .collect();

        let mut point_buf: UnifiedBuffer<GpuPoint3D> =
            UnifiedBuffer::new(ctx.device(), gpu_points.len())?;
        point_buf.write(&gpu_points);

        let aspect = target.width() as f32 / target.height() as f32;
        let uniforms = PointRenderUniforms {
            mvp: camera.mvp_matrix(aspect),
            point_size,
            _pad: [0.0; 3],
        };

        let _pg = point_buf.gpu_guard();

        let cmd_buf = ctx
            .queue()
            .commandBuffer()
            .ok_or(Error::Gpu("failed to create command buffer".into()))?;

        let pass_desc = MTLRenderPassDescriptor::new();
        let color_attachment = unsafe { pass_desc.colorAttachments().objectAtIndexedSubscript(0) };
        color_attachment.setTexture(Some(target.color_texture().raw()));
        color_attachment.setLoadAction(MTLLoadAction::Clear);
        color_attachment.setStoreAction(MTLStoreAction::Store);
        color_attachment.setClearColor(MTLClearColor {
            red: 0.0,
            green: 0.0,
            blue: 0.0,
            alpha: 1.0,
        });

        let depth_attachment = pass_desc.depthAttachment();
        depth_attachment.setTexture(Some(target.depth_raw()));
        depth_attachment.setLoadAction(MTLLoadAction::Clear);
        depth_attachment.setStoreAction(MTLStoreAction::DontCare);
        depth_attachment.setClearDepth(1.0);

        let encoder = cmd_buf
            .renderCommandEncoderWithDescriptor(&pass_desc)
            .ok_or(Error::Gpu("failed to create render encoder".into()))?;

        unsafe {
            encoder.setRenderPipelineState(&self.pipeline);
            encoder.setVertexBuffer_offset_atIndex(Some(point_buf.metal_buffer()), 0, 0);
            encoder.setVertexBytes_length_atIndex(
                NonNull::new_unchecked(&uniforms as *const PointRenderUniforms as *mut c_void),
                mem::size_of::<PointRenderUniforms>(),
                1,
            );
            encoder.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Point, 0, cloud.len());
        }

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        drop(_pg);
        Ok(())
    }
}
