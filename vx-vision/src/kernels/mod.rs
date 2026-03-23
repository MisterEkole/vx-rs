//! GPU compute kernel bindings.

pub mod fast;
pub mod gaussian;
pub mod harris;
pub mod klt;
pub mod nms;
pub mod orb;
pub mod stereomatch;
pub mod undistort;

pub mod canny;
pub mod color;
pub mod dense_flow;
pub mod histogram;
pub mod homography;
pub mod integral;
pub mod morphology;
pub mod pyramid;
pub mod resize;
pub mod sobel;
pub mod threshold;
pub mod warp;

pub mod bilateral;
pub mod connected;
pub mod distance;
pub mod dog;
pub mod hough;
pub mod indirect;
pub mod matcher;
pub mod sift;
pub mod template_match;

// ── 3D Reconstruction kernels ──
#[cfg(feature = "reconstruction")]
pub mod depth_colorize;
#[cfg(feature = "reconstruction")]
pub mod depth_filter;
#[cfg(feature = "reconstruction")]
pub mod depth_inpaint;
#[cfg(feature = "reconstruction")]
pub mod depth_to_cloud;
#[cfg(feature = "reconstruction")]
pub mod marching_cubes;
#[cfg(feature = "reconstruction")]
pub mod normal_estimation;
#[cfg(feature = "reconstruction")]
pub mod outlier_filter;
#[cfg(feature = "reconstruction")]
pub mod sgm;
#[cfg(feature = "reconstruction")]
pub mod triangulate;
#[cfg(feature = "reconstruction")]
pub mod tsdf;
#[cfg(feature = "reconstruction")]
pub mod voxel_downsample;
