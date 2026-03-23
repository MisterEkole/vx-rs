//! Metal device initialization and shader library loading.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary};

#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {}

/// Returns the system default Metal device, or `None` if Metal is unavailable.
pub fn default_device() -> Option<Retained<ProtocolObject<dyn MTLDevice>>> {
    MTLCreateSystemDefaultDevice()
}

/// Creates a command queue on `device`.
pub fn new_queue(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Result<Retained<ProtocolObject<dyn MTLCommandQueue>>, String> {
    device
        .newCommandQueue()
        .ok_or_else(|| "Failed to create command queue".into())
}

/// Loads a Metal library from precompiled metallib bytes via a temp file.
pub fn load_library_from_bytes(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: &[u8],
) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, String> {
    if bytes.is_empty() {
        return Err("Metallib bytes are empty".into());
    }

    let tmp_dir = std::env::temp_dir();
    let unique_id = std::process::id();
    let thread_id = format!("{:?}", std::thread::current().id());
    let metallib_path = tmp_dir.join(format!(
        "vx_shaders_{}_{}.metallib",
        unique_id,
        thread_id.replace(['(', ')'], "")
    ));
    std::fs::write(&metallib_path, bytes)
        .map_err(|e| format!("Failed to write metallib to temp: {e}"))?;

    let url_string =
        objc2_foundation::NSString::from_str(&format!("file://{}", metallib_path.display()));
    let url = objc2_foundation::NSURL::URLWithString(&url_string)
        .ok_or("Failed to create NSURL for metallib")?;

    let library = device
        .newLibraryWithURL_error(&url)
        .map_err(|e| format!("Failed to load metallib: {e}"))?;

    let _ = std::fs::remove_file(&metallib_path);

    Ok(library)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_device_exists() {
        let device = default_device();
        assert!(
            device.is_some(),
            "No Metal device found — Apple Silicon or compatible GPU required"
        );
    }

    #[test]
    fn create_queue() {
        let device = default_device().expect("No Metal device");
        let queue = new_queue(&device);
        assert!(
            queue.is_ok(),
            "Failed to create command queue: {:?}",
            queue.err()
        );
    }

    #[test]
    fn load_empty_metallib_fails() {
        let device = default_device().expect("No Metal device");
        let result = load_library_from_bytes(&device, &[]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty"));
    }

    #[test]
    fn load_invalid_metallib_fails() {
        let device = default_device().expect("No Metal device");
        let result = load_library_from_bytes(&device, b"not a metallib");
        assert!(result.is_err());
    }
}
