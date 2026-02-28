// vx-core/src/lib.rs
//
// Memory management and Metal wrappers for VX.

pub mod buffer;
pub mod device;

// Re-export commonly used types at crate root
pub use buffer::{GpuGuard, UnifiedBuffer};
pub use device::{default_device, load_library_from_bytes, new_queue};