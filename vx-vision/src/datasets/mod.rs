//! Dataset loaders for standard computer vision benchmarks.
//!
//! Loaders return metadata and file paths. Image loading is the caller's
//! responsibility (use the `image` crate + `Context::texture_*()` methods).

pub mod euroc;
pub mod kitti;
pub mod tum;

use std::path::PathBuf;

/// A single frame from a dataset.
#[derive(Clone, Debug)]
pub struct DatasetFrame {
    /// Timestamp in seconds.
    pub timestamp: f64,
    /// Path to the RGB or grayscale image file.
    pub image_path: PathBuf,
    /// Path to the depth image (if available).
    pub depth_path: Option<PathBuf>,
    /// Camera pose (if available). World-to-camera transform.
    #[cfg(feature = "reconstruction")]
    pub pose: Option<crate::types_3d::CameraExtrinsics>,
}

/// Iterator over dataset frames.
pub trait DatasetIterator: Iterator<Item = DatasetFrame> {
    /// Returns the total number of frames (if known).
    fn len(&self) -> Option<usize>;
}
