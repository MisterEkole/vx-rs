// vx-vision/src/context.rs
//
// The user-facing GPU context. Initializes Metal device, command queue,
// and loads the embedded shader library in one call.
//
// This is the main entry point — users should never need to import
// objc2 or objc2-metal types.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandQueue, MTLDevice, MTLLibrary};

use crate::texture::Texture;

/// GPU context holding a Metal device, command queue, and compiled shader library.
///
/// Create once at startup and reuse for the lifetime of your application.
///
/// # Example
/// ```no_run
/// let ctx = vx_vision::Context::new().expect("No Metal GPU available");
/// ```
pub struct Context {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
}

impl Context {
    /// Initialize the default Metal device and load the VX shader library.
    ///
    /// Fails if no Metal-capable GPU is found or if the embedded shaders
    /// cannot be loaded.
    pub fn new() -> Result<Self, String> {
        let device = vx_core::default_device()
            .ok_or_else(|| "No Metal device found (Apple Silicon or discrete GPU required)".to_string())?;
        let queue = vx_core::new_queue(&device)?;
        let library = crate::load_library(&device)?;

        Ok(Self { device, queue, library })
    }

    /// Create a single-channel 8-bit grayscale texture.
    ///
    /// `pixels` must contain exactly `width * height` bytes in row-major order.
    ///
    /// # Example
    /// ```no_run
    /// # let ctx = vx_vision::Context::new().unwrap();
    /// let img = image::open("photo.png").unwrap().to_luma8();
    /// let (w, h) = img.dimensions();
    /// let texture = ctx.texture_gray8(img.as_raw(), w, h).unwrap();
    /// ```
    pub fn texture_gray8(
        &self,
        pixels: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Texture, String> {
        Texture::from_gray8(&self.device, pixels, width, height)
    }

    /// Create a single-channel R32Float texture from f32 remap data.
    ///
    /// Use this to upload `map_x` / `map_y` arrays produced by camera
    /// calibration (e.g. OpenCV's `initUndistortRectifyMap`).
    /// `data` must contain exactly `width * height` values.
    pub fn texture_r32float(
        &self,
        data:   &[f32],
        width:  u32,
        height: u32,
    ) -> Result<Texture, String> {
        Texture::from_r32float(&self.device, data, width, height)
    }

    /// Create a write-capable R8Unorm output texture for the undistort kernel.
    ///
    /// After the kernel completes, call [`Texture::read_gray8`] to copy
    /// the result back to the CPU.
    pub fn texture_output_gray8(&self, width: u32, height: u32) -> Result<Texture, String> {
        Texture::output_gray8(&self.device, width, height)
    }

    // -- Crate-internal accessors (not part of the public API) --

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