// vx-vision/src/types.rs
//
// CPU-side mirrors of every MSL struct used across kernels.
// All types are #[repr(C)] + Pod so they can be blitted straight
// into Metal shared-memory buffers with zero conversion.

use bytemuck::{Pod, Zeroable};

/// Matches `CornerPoint` in every .metal file.
/// Layout: float2 position (8B) + float response (4B) + uint pyramid_level (4B) = 16B
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct CornerPoint {
    pub position: [f32; 2],
    pub response: f32,
    pub pyramid_level: u32,
}

/// Matches `FASTParams` in FastDetect.metal.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct FASTParams {
    pub threshold: i32,
    pub max_corners: u32,
    pub width: u32,
    pub height: u32,
}

/// Matches `HarrisParams` in HarrisResponse.metal.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct HarrisParams {
    pub n_corners: u32,
    pub patch_radius: i32,
    pub k: f32,
}

/// Matches `ORBOutput` in ORBDescriptor.metal.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ORBOutput {
    pub desc: [u32; 8],  // 256-bit descriptor
    pub angle: f32,
}

/// Matches `ORBParams` in ORBDescriptor.metal.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ORBParams {
    pub n_keypoints: u32,
    pub patch_radius: u32,
}

/// Matches `KLTParams` in KLTTracker.metal.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct KLTParams {
    pub n_points: u32,
    pub max_iterations: u32,
    pub epsilon: f32,
    pub win_radius: i32,
    pub max_level: u32,
    pub min_eigenvalue: f32,
}

/// Matches `StereoMatchResult` in StereoMatch.metal.
/// Note: MSL float3 is 16-byte aligned, so we pad to match.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct StereoMatchResult {
    pub left_idx: u32,
    pub right_idx: u32,
    pub disparity: f32,
    pub point_3d: [f32; 3],
    pub _pad: f32,
}

/// Matches `StereoParams` in StereoMatch.metal.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct StereoParams {
    pub n_left: u32,
    pub n_right: u32,
    pub max_epipolar: f32,
    pub min_disparity: f32,
    pub max_disparity: f32,
    pub max_hamming: u32,
    pub ratio_thresh: f32,
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub baseline: f32,
}