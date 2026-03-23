//! Application entry point for GPU initialization.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandQueue, MTLDevice, MTLLibrary};

use crate::error::{Error, Result};
use crate::texture::Texture;

/// Metal device, command queue, and compiled shader library.
///
/// # Example
///
/// ```no_run
/// let ctx = vx_vision::Context::new().expect("No Metal GPU available");
/// ```
pub struct Context {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
}

impl Context {
    /// Initializes the default Metal device and loads the embedded shader library.
    #[must_use = "contains the GPU context needed for all operations"]
    pub fn new() -> Result<Self> {
        let device = vx_gpu::default_device()
            .ok_or(Error::DeviceNotFound)?;
        let queue = vx_gpu::new_queue(&device)?;
        let library = crate::load_library(&device)?;

        Ok(Self { device, queue, library })
    }

    /// Creates an R8Unorm texture from grayscale pixel data.
    pub fn texture_gray8(
       
        &self,
        pixels: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Texture> {
        Texture::from_gray8(&self.device, pixels, width, height)
    }

    /// Creates an R32Float texture from `f32` data.
    pub fn texture_r32float(
        &self,
        data:   &[f32],
        width:  u32,
        height: u32,
    ) -> Result<Texture> {
        Texture::from_r32float(&self.device, data, width, height)
    }

    /// Creates an empty R8Unorm output texture with `ShaderWrite` usage.
    pub fn texture_output_gray8(&self, width: u32, height: u32) -> Result<Texture> {
        Texture::output_gray8(&self.device, width, height)
    }

    /// Creates an RGBA8Unorm texture from 4-channel pixel data.
    pub fn texture_rgba8(
        &self,
        pixels: &[u8],
        width:  u32,
        height: u32,
    ) -> Result<Texture> {
        Texture::from_rgba8(&self.device, pixels, width, height)
    }

    /// Creates an empty RGBA8Unorm output texture.
    pub fn texture_output_rgba8(&self, width: u32, height: u32) -> Result<Texture> {
        Texture::output_rgba8(&self.device, width, height)
    }

    /// Creates an empty R32Float output texture.
    pub fn texture_output_r32float(&self, width: u32, height: u32) -> Result<Texture> {
        Texture::output_r32float(&self.device, width, height)
    }

    /// Creates an R8Unorm texture with `ShaderRead | ShaderWrite` usage for pipeline intermediates.
    pub fn texture_intermediate_gray8(&self, width: u32, height: u32) -> Result<Texture> {
        Texture::intermediate_gray8(&self.device, width, height)
    }

    /// Creates an R32Float texture with `ShaderRead | ShaderWrite` usage for pipeline intermediates.
    pub fn texture_intermediate_r32float(&self, width: u32, height: u32) -> Result<Texture> {
        Texture::intermediate_r32float(&self.device, width, height)
    }

    pub(crate) fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    pub(crate) fn queue(&self) -> &ProtocolObject<dyn MTLCommandQueue> {
        &self.queue
    }

    pub(crate) fn library(&self) -> &ProtocolObject<dyn MTLLibrary> {
        &self.library
    }
}

// SAFETY: MTLDevice, MTLCommandQueue, and MTLLibrary are all thread-safe.
unsafe impl Send for Context {}
unsafe impl Sync for Context {}