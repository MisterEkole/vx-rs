//! Mesh renderer using Metal render pipeline with Phong shading.

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
use vx_gpu::UnifiedBuffer;

#[cfg(feature = "reconstruction")]
use crate::types_3d::Mesh;

/// GPU vertex for mesh rendering (matches MeshVertex in shader).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuMeshVertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
}

/// GPU uniforms for mesh rendering.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct MeshRenderUniforms {
    mvp: [f32; 16],
    model: [f32; 16],
    light_dir: [f32; 4],
    light_color: [f32; 4],
    ambient: [f32; 4],
}

/// Compiled mesh render pipeline.
pub struct MeshRenderer {
    pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
}

unsafe impl Send for MeshRenderer {}
unsafe impl Sync for MeshRenderer {}

impl MeshRenderer {
    /// Compiles the mesh render pipeline.
    pub fn new(ctx: &Context) -> Result<Self> {
        let vertex_func = ctx
            .library()
            .newFunctionWithName(objc2_foundation::ns_string!("mesh_vertex"))
            .ok_or(Error::ShaderMissing("mesh_vertex".into()))?;
        let fragment_func = ctx
            .library()
            .newFunctionWithName(objc2_foundation::ns_string!("mesh_fragment"))
            .ok_or(Error::ShaderMissing("mesh_fragment".into()))?;

        let desc = MTLRenderPipelineDescriptor::new();
        desc.setVertexFunction(Some(&vertex_func));
        desc.setFragmentFunction(Some(&fragment_func));

        let color_attachment = unsafe { desc.colorAttachments().objectAtIndexedSubscript(0) };
        color_attachment.setPixelFormat(MTLPixelFormat::RGBA8Unorm);

        desc.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float);

        let pipeline = ctx
            .device()
            .newRenderPipelineStateWithDescriptor_error(&desc)
            .map_err(|e| Error::PipelineCompile(format!("mesh_render: {e}")))?;

        Ok(Self { pipeline })
    }

    /// Renders a mesh to the render target with Phong shading. Synchronous.
    #[cfg(feature = "reconstruction")]
    pub fn render(
        &self,
        ctx: &Context,
        mesh: &Mesh,
        camera: &Camera,
        target: &RenderTarget,
    ) -> Result<()> {
        if mesh.is_empty() {
            return Ok(());
        }

        // Flatten indexed mesh into a triangle list
        let mut vertices = Vec::with_capacity(mesh.indices.len() * 3);
        for tri in &mesh.indices {
            for &idx in tri {
                let v = &mesh.vertices[idx as usize];
                vertices.push(GpuMeshVertex {
                    position: v.position,
                    normal: v.normal,
                    uv: v.uv,
                });
            }
        }

        let mut vertex_buf: UnifiedBuffer<GpuMeshVertex> =
            UnifiedBuffer::new(ctx.device(), vertices.len())?;
        vertex_buf.write(&vertices);

        let aspect = target.width() as f32 / target.height() as f32;
        let identity = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let uniforms = MeshRenderUniforms {
            mvp: camera.mvp_matrix(aspect),
            model: identity,
            light_dir: [0.5, 0.7, 1.0, 0.0],
            light_color: [0.8, 0.8, 0.8, 1.0],
            ambient: [0.2, 0.2, 0.2, 1.0],
        };

        let _vg = vertex_buf.gpu_guard();

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
            red: 0.1,
            green: 0.1,
            blue: 0.15,
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
            encoder.setVertexBuffer_offset_atIndex(Some(vertex_buf.metal_buffer()), 0, 0);
            encoder.setVertexBytes_length_atIndex(
                NonNull::new_unchecked(&uniforms as *const MeshRenderUniforms as *mut c_void),
                mem::size_of::<MeshRenderUniforms>(),
                1,
            );
            encoder.setFragmentBytes_length_atIndex(
                NonNull::new_unchecked(&uniforms as *const MeshRenderUniforms as *mut c_void),
                mem::size_of::<MeshRenderUniforms>(),
                1,
            );
            encoder.drawPrimitives_vertexStart_vertexCount(
                MTLPrimitiveType::Triangle,
                0,
                vertices.len(),
            );
        }

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        drop(_vg);
        Ok(())
    }
}
