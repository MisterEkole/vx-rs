// vx-core/src/device.rs
//
// Device and Queue initialization helpers.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
};

// Required for MTLCreateSystemDefaultDevice
#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {}

/// Get the system default Metal device.
/// Returns `None` on machines without Metal support.
pub fn default_device() -> Option<Retained<ProtocolObject<dyn MTLDevice>>> {
    MTLCreateSystemDefaultDevice()
}

/// Create a new command queue from a device.
pub fn new_queue(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Result<Retained<ProtocolObject<dyn MTLCommandQueue>>, String> {
    device.newCommandQueue().ok_or_else(|| "Failed to create command queue".into())
}

/// Load a Metal library from raw metallib bytes.
///
/// Writes to a temp file and loads via `newLibraryWithURL:error:`.
/// The bytes are typically embedded at compile time via `include_bytes!`.
pub fn load_library_from_bytes(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: &[u8],
) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, String> {
    if bytes.is_empty() {
        return Err("Metallib bytes are empty".into());
    }

    let tmp_dir = std::env::temp_dir();
    let metallib_path = tmp_dir.join("vx_shaders.metallib");
    std::fs::write(&metallib_path, bytes)
        .map_err(|e| format!("Failed to write metallib to temp: {e}"))?;

    let url_string =
        objc2_foundation::NSString::from_str(&format!("file://{}", metallib_path.display()));
    let url = objc2_foundation::NSURL::URLWithString(&url_string)
        .ok_or("Failed to create NSURL for metallib")?;

    let library = device.newLibraryWithURL_error(&url)
        .map_err(|e| format!("Failed to load metallib: {e}"))?;

    let _ = std::fs::remove_file(&metallib_path);

    Ok(library)
}