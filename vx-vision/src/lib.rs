// vx-vision/src/lib.rs
//
// High-level CV algorithms for VX.
//
// Users should only need to import from this crate.
// The objc2 / objc2-metal types are kept internal.

pub mod context;
pub mod kernels;
pub mod texture;
pub mod types;

// Primary public API
pub use context::Context;
pub use texture::Texture;

// ── Internal ──

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLDevice, MTLLibrary};

/// Embedded metallib compiled at build time by build.rs.
static METALLIB_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/vx.metallib"));

/// Load the VX shader library from embedded bytes (crate-internal).
pub(crate) fn load_library(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, String> {
    vx_core::load_library_from_bytes(device, METALLIB_BYTES)
}