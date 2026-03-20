//! GPU-accelerated computer vision kernels for Apple Silicon.
//!
//! All Metal internals are hidden behind [`Context`], [`Texture`], and the
//! kernel structs in [`kernels`]. Users never need to import `objc2-metal`.

pub mod context;
pub mod kernels;
pub mod pipeline;
pub mod pool;
pub mod texture;
pub mod types;

pub use context::Context;
pub use pipeline::Pipeline;
pub use pool::TexturePool;
pub use texture::{Texture, TextureFormat};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLDevice, MTLLibrary};

static METALLIB_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/vx.metallib"));

/// Loads the embedded VX shader library onto `device`.
pub(crate) fn load_library(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, String> {
    vx_gpu::load_library_from_bytes(device, METALLIB_BYTES)
}