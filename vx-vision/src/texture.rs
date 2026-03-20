//! GPU texture wrapper with format-aware readback.

use core::ffi::c_void;
use core::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLDevice, MTLOrigin, MTLPixelFormat, MTLRegion, MTLSize,
    MTLTexture as MTLTextureTrait, MTLTextureDescriptor, MTLTextureUsage,
};

/// Pixel format of a [`Texture`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureFormat {
    /// Single-channel 8-bit normalized unsigned integer.
    R8Unorm,
    /// Single-channel 32-bit float.
    R32Float,
    /// Four-channel 8-bit normalized unsigned integer.
    RGBA8Unorm,
}

/// A Metal texture with tracked dimensions and format.
///
/// Construct via [`Context`](crate::Context) methods or [`Texture::from_metal_texture`].
pub struct Texture {
    raw: Retained<ProtocolObject<dyn MTLTextureTrait>>,
    width: u32,
    height: u32,
    format: TextureFormat,
}

impl Texture {
    /// Creates an R8Unorm texture from `width * height` bytes.
    pub(crate) fn from_gray8(
        device: &ProtocolObject<dyn MTLDevice>,
        pixels: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        let expected = (width as usize) * (height as usize);
        if pixels.len() != expected {
            return Err(format!(
                "Expected {} bytes for {}x{} R8 texture, got {}",
                expected, width, height, pixels.len()
            ));
        }

        let desc = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat::R8Unorm,
                width as usize,
                height as usize,
                false,
            )
        };
        desc.setUsage(MTLTextureUsage::ShaderRead);

        let raw = device
            .newTextureWithDescriptor(&desc)
            .ok_or("Failed to create Metal texture")?;

        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width: width as usize,
                height: height as usize,
                depth: 1,
            },
        };

        unsafe {
            raw.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                region,
                0,
                NonNull::new_unchecked(pixels.as_ptr() as *mut c_void),
                width as usize,
            );
        }

        Ok(Self { raw, width, height, format: TextureFormat::R8Unorm })
    }

    /// Creates an R32Float texture from `width * height` floats.
    pub(crate) fn from_r32float(
        device: &ProtocolObject<dyn MTLDevice>,
        data:   &[f32],
        width:  u32,
        height: u32,
    ) -> Result<Self, String> {
        let expected = (width as usize) * (height as usize);
        if data.len() != expected {
            return Err(format!(
                "Expected {} f32 values for {}x{} R32Float texture, got {}",
                expected, width, height, data.len()
            ));
        }

        let desc = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat::R32Float,
                width as usize,
                height as usize,
                false,
            )
        };
        desc.setUsage(MTLTextureUsage::ShaderRead);

        let raw = device
            .newTextureWithDescriptor(&desc)
            .ok_or("Failed to create R32Float Metal texture")?;

        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width:  width as usize,
                height: height as usize,
                depth:  1,
            },
        };

        unsafe {
            raw.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                region,
                0,
                NonNull::new_unchecked(data.as_ptr() as *mut c_void),
                width as usize * size_of::<f32>(),
            );
        }

        Ok(Self { raw, width, height, format: TextureFormat::R32Float })
    }

    /// Creates an empty R8Unorm texture with `ShaderWrite` usage.
    pub(crate) fn output_gray8(
        device: &ProtocolObject<dyn MTLDevice>,
        width:  u32,
        height: u32,
    ) -> Result<Self, String> {
        let desc = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat::R8Unorm,
                width as usize,
                height as usize,
                false,
            )
        };
        desc.setUsage(MTLTextureUsage::ShaderWrite);

        let raw = device
            .newTextureWithDescriptor(&desc)
            .ok_or("Failed to create output Metal texture")?;

        Ok(Self { raw, width, height, format: TextureFormat::R8Unorm })
    }

    /// Creates an R32Float texture with `ShaderRead | ShaderWrite` usage.
    pub(crate) fn intermediate_r32float(
        device: &ProtocolObject<dyn MTLDevice>,
        width:  u32,
        height: u32,
    ) -> Result<Self, String> {
        let desc = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat::R32Float,
                width as usize,
                height as usize,
                false,
            )
        };
        desc.setUsage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);

        let raw = device
            .newTextureWithDescriptor(&desc)
            .ok_or("Failed to create intermediate R32Float texture")?;

        Ok(Self { raw, width, height, format: TextureFormat::R32Float })
    }

    /// Creates an R8Unorm texture with `ShaderRead | ShaderWrite` usage.
    pub(crate) fn intermediate_gray8(
        device: &ProtocolObject<dyn MTLDevice>,
        width:  u32,
        height: u32,
    ) -> Result<Self, String> {
        let desc = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat::R8Unorm,
                width as usize,
                height as usize,
                false,
            )
        };
        desc.setUsage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);

        let raw = device
            .newTextureWithDescriptor(&desc)
            .ok_or("Failed to create intermediate R8Unorm texture")?;

        Ok(Self { raw, width, height, format: TextureFormat::R8Unorm })
    }

    /// Creates an RGBA8Unorm texture from `width * height * 4` bytes.
    pub(crate) fn from_rgba8(
        device: &ProtocolObject<dyn MTLDevice>,
        pixels: &[u8],
        width:  u32,
        height: u32,
    ) -> Result<Self, String> {
        let expected = (width as usize) * (height as usize) * 4;
        if pixels.len() != expected {
            return Err(format!(
                "Expected {} bytes for {}x{} RGBA8 texture, got {}",
                expected, width, height, pixels.len()
            ));
        }

        let desc = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat::RGBA8Unorm,
                width as usize,
                height as usize,
                false,
            )
        };
        desc.setUsage(MTLTextureUsage::ShaderRead);

        let raw = device
            .newTextureWithDescriptor(&desc)
            .ok_or("Failed to create RGBA8 Metal texture")?;

        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width:  width as usize,
                height: height as usize,
                depth:  1,
            },
        };

        unsafe {
            raw.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                region,
                0,
                NonNull::new_unchecked(pixels.as_ptr() as *mut c_void),
                width as usize * 4,
            );
        }

        Ok(Self { raw, width, height, format: TextureFormat::RGBA8Unorm })
    }

    /// Creates an empty RGBA8Unorm texture with `ShaderRead | ShaderWrite` usage.
    pub(crate) fn output_rgba8(
        device: &ProtocolObject<dyn MTLDevice>,
        width:  u32,
        height: u32,
    ) -> Result<Self, String> {
        let desc = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat::RGBA8Unorm,
                width as usize,
                height as usize,
                false,
            )
        };
        desc.setUsage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);

        let raw = device
            .newTextureWithDescriptor(&desc)
            .ok_or("Failed to create output RGBA8 texture")?;

        Ok(Self { raw, width, height, format: TextureFormat::RGBA8Unorm })
    }

    /// Creates an empty R32Float texture with `ShaderWrite` usage.
    pub(crate) fn output_r32float(
        device: &ProtocolObject<dyn MTLDevice>,
        width:  u32,
        height: u32,
    ) -> Result<Self, String> {
        let desc = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat::R32Float,
                width as usize,
                height as usize,
                false,
            )
        };
        desc.setUsage(MTLTextureUsage::ShaderWrite);

        let raw = device
            .newTextureWithDescriptor(&desc)
            .ok_or("Failed to create output R32Float texture")?;

        Ok(Self { raw, width, height, format: TextureFormat::R32Float })
    }

    /// Reads back the texture contents as `width * height` R8 bytes.
    ///
    /// Must be called after the writing command buffer has completed.
    pub fn read_gray8(&self) -> Vec<u8> {
        let n = (self.width as usize) * (self.height as usize);
        let mut buf = vec![0u8; n];

        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width:  self.width as usize,
                height: self.height as usize,
                depth:  1,
            },
        };

        unsafe {
            self.raw.getBytes_bytesPerRow_fromRegion_mipmapLevel(
                NonNull::new_unchecked(buf.as_mut_ptr() as *mut c_void),
                self.width as usize,
                region,
                0,
            );
        }

        buf
    }

    /// Reads back the texture contents as `width * height * 4` RGBA bytes.
    pub fn read_rgba8(&self) -> Vec<u8> {
        let n = (self.width as usize) * (self.height as usize) * 4;
        let mut buf = vec![0u8; n];

        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width:  self.width as usize,
                height: self.height as usize,
                depth:  1,
            },
        };

        unsafe {
            self.raw.getBytes_bytesPerRow_fromRegion_mipmapLevel(
                NonNull::new_unchecked(buf.as_mut_ptr() as *mut c_void),
                self.width as usize * 4,
                region,
                0,
            );
        }

        buf
    }

    /// Reads back the texture contents as `width * height` floats.
    pub fn read_r32float(&self) -> Vec<f32> {
        let n = (self.width as usize) * (self.height as usize);
        let mut buf = vec![0.0f32; n];

        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width:  self.width as usize,
                height: self.height as usize,
                depth:  1,
            },
        };

        unsafe {
            self.raw.getBytes_bytesPerRow_fromRegion_mipmapLevel(
                NonNull::new_unchecked(buf.as_mut_ptr() as *mut c_void),
                self.width as usize * size_of::<f32>(),
                region,
                0,
            );
        }

        buf
    }

    /// Returns the width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Returns the height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Returns the pixel format.
    pub fn format(&self) -> TextureFormat {
        self.format
    }

    pub(crate) fn raw(&self) -> &ProtocolObject<dyn MTLTextureTrait> {
        &self.raw
    }

    /// Wraps an existing `MTLTexture` without copying data.
    ///
    /// Intended for zero-copy AVFoundation / Core Video integration.
    /// The caller must ensure the underlying texture outlives this wrapper.
    pub fn from_metal_texture(
        raw:    Retained<ProtocolObject<dyn MTLTextureTrait>>,
        width:  u32,
        height: u32,
        format: TextureFormat,
    ) -> Self {
        Self { raw, width, height, format }
    }
}
