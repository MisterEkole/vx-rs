//! Render target and render pipeline utilities for visualization.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLDevice, MTLPixelFormat, MTLTexture as MTLTextureTrait, MTLTextureDescriptor, MTLTextureUsage,
};

use crate::context::Context;
use crate::error::{Error, Result};
use crate::texture::Texture;

/// An offscreen render target with color and depth attachments.
pub struct RenderTarget {
    color: Texture,
    depth: Retained<ProtocolObject<dyn MTLTextureTrait>>,
    width: u32,
    height: u32,
}

impl RenderTarget {
    /// Creates a new offscreen render target with RGBA8 color and Depth32Float.
    pub fn new(ctx: &Context, width: u32, height: u32) -> Result<Self> {
        let color = Texture::output_rgba8(ctx.device(), width, height)?;

        // Create depth texture
        let depth_desc = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat::Depth32Float,
                width as usize,
                height as usize,
                false,
            )
        };
        depth_desc.setUsage(MTLTextureUsage::RenderTarget);

        let depth = ctx
            .device()
            .newTextureWithDescriptor(&depth_desc)
            .ok_or(Error::Gpu("failed to create depth texture".into()))?;

        Ok(Self {
            color,
            depth,
            width,
            height,
        })
    }

    /// Returns the color texture (RGBA8Unorm).
    pub fn color_texture(&self) -> &Texture {
        &self.color
    }

    /// Returns the raw depth Metal texture.
    pub(crate) fn depth_raw(&self) -> &ProtocolObject<dyn MTLTextureTrait> {
        &self.depth
    }

    /// Reads back the rendered image as RGBA bytes.
    pub fn read_rgba8(&self) -> Vec<u8> {
        self.color.read_rgba8()
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }
}

/// Camera parameters for rendering.
#[derive(Clone, Debug)]
pub struct Camera {
    pub position: [f32; 3],
    pub look_at: [f32; 3],
    pub up: [f32; 3],
    pub fov_y: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 3.0],
            look_at: [0.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            fov_y: 60.0_f32.to_radians(),
            near: 0.01,
            far: 100.0,
        }
    }
}

impl Camera {
    /// Computes the model-view-projection matrix as a column-major [f32; 16].
    pub fn mvp_matrix(&self, aspect: f32) -> [f32; 16] {
        let view = look_at(self.position, self.look_at, self.up);
        let proj = perspective(self.fov_y, aspect, self.near, self.far);
        mat4_mul(&proj, &view)
    }
}

// Column-major 4x4 matrix helpers

fn look_at(eye: [f32; 3], center: [f32; 3], up: [f32; 3]) -> [f32; 16] {
    let f = normalize3([center[0] - eye[0], center[1] - eye[1], center[2] - eye[2]]);
    let s = normalize3(cross3(f, up));
    let u = cross3(s, f);

    [
        s[0],
        u[0],
        -f[0],
        0.0,
        s[1],
        u[1],
        -f[1],
        0.0,
        s[2],
        u[2],
        -f[2],
        0.0,
        -dot3(s, eye),
        -dot3(u, eye),
        dot3(f, eye),
        1.0,
    ]
}

fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> [f32; 16] {
    let f = 1.0 / (fov_y / 2.0).tan();
    let nf = 1.0 / (near - far);
    [
        f / aspect,
        0.0,
        0.0,
        0.0,
        0.0,
        f,
        0.0,
        0.0,
        0.0,
        0.0,
        (far + near) * nf,
        -1.0,
        0.0,
        0.0,
        2.0 * far * near * nf,
        0.0,
    ]
}

fn mat4_mul(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut result = [0.0f32; 16];
    for col in 0..4 {
        for row in 0..4 {
            let mut sum = 0.0;
            for k in 0..4 {
                sum += a[row + k * 4] * b[k + col * 4];
            }
            result[row + col * 4] = sum;
        }
    }
    result
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 1e-10 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 0.0, 1.0]
    }
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
