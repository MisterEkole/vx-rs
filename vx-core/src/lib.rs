//! Shared-memory Metal buffer management for Apple Silicon UMA.

pub mod buffer;
pub mod device;

pub use buffer::{GpuGuard, UnifiedBuffer};
pub use device::{default_device, load_library_from_bytes, new_queue};