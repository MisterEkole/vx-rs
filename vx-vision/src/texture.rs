// vx-vision/src/texture.rs
//
// Public Texture wrapper. Users create textures through Context methods,
// never touching objc2-metal types directly.

use core::ffi::c_void;
use core::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLDevice, MTLOrigin, MTLPixelFormat, MTLRegion, MTLSize,
    MTLTexture as MTLTextureTrait, MTLTextureDescriptor, MTLTextureUsage,
};

/// A GPU texture backed by Metal.
///
/// Users should not construct this directly — use
/// [`Context::texture_gray8`](crate::Context::texture_gray8) instead.
pub struct Texture {
    raw: Retained<ProtocolObject<dyn MTLTextureTrait>>,
    width: u32,
    height: u32,
}

impl Texture {
    /// Create a single-channel 8-bit grayscale texture from raw pixel bytes.
    ///
    /// `pixels` must contain exactly `width * height` bytes in row-major order.
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

        // SAFETY: pixels is a valid slice with the correct byte count.
        unsafe {
            raw.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                region,
                0,
                NonNull::new_unchecked(pixels.as_ptr() as *mut c_void),
                width as usize,
            );
        }

        Ok(Self { raw, width, height })
    }

    /// Create a single-channel R32Float texture from a slice of f32 values.
    ///
    /// Used for undistortion remap maps (map_x / map_y). Each float is the
    /// source pixel coordinate for the corresponding output pixel.
    /// `data` must contain exactly `width * height` f32 values.
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

        // SAFETY: data is a valid slice with the correct element count.
        unsafe {
            raw.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                region,
                0,
                NonNull::new_unchecked(data.as_ptr() as *mut c_void),
                width as usize * size_of::<f32>(),
            );
        }

        Ok(Self { raw, width, height })
    }

    /// Create a write-capable R8Unorm output texture (no initial data).
    ///
    /// Used as the destination for the undistort kernel. After the GPU
    /// has finished, call [`Self::read_gray8`] to copy pixels back to CPU.
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

        Ok(Self { raw, width, height })
    }

    /// Copy the texture contents back to CPU as raw R8 bytes.
    ///
    /// **Only call after the command buffer that wrote this texture has
    /// completed.** Returns `width * height` bytes in row-major order.
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

        // SAFETY: buf has exactly n bytes matching the texture dimensions.
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

    /// Width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Access the underlying Metal texture (crate-internal only).
    pub(crate) fn raw(&self) -> &ProtocolObject<dyn MTLTextureTrait> {
        &self.raw
    }
}