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