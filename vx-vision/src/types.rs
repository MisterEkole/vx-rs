//! `#[repr(C)]` mirrors of Metal shader structs for zero-copy buffer binding.

use bytemuck::{Pod, Zeroable};

/// Detected corner: position, response score, and pyramid level.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct CornerPoint {
    pub position: [f32; 2],
    pub response: f32,
    pub pyramid_level: u32,
}

/// GPU parameters for FAST-9 corner detection.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct FASTParams {
    pub threshold: i32,
    pub max_corners: u32,
    pub width: u32,
    pub height: u32,
}

/// GPU parameters for Harris corner response.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct HarrisParams {
    pub n_corners: u32,
    pub patch_radius: i32,
    pub k: f32,
}

/// ORB descriptor output: 256-bit descriptor and orientation angle.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ORBOutput {
    pub desc: [u32; 8],  // 256-bit descriptor
    pub angle: f32,
}

/// GPU parameters for ORB descriptor computation.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ORBParams {
    pub n_keypoints: u32,
    pub patch_radius: u32,
}

/// GPU parameters for KLT optical flow tracking.
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

/// Stereo match result with disparity and 3D point. Padded to match MSL `float3` alignment.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct StereoMatchResult {
    pub left_idx: u32,
    pub right_idx: u32,
    pub disparity: f32,
    pub point_3d: [f32; 3],
    pub _pad: f32,
}

/// GPU parameters for stereo matching.
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

/// GPU parameters for non-maximum suppression.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct NMSParams {
    pub n_corners:    u32,
    pub min_distance: f32,
}

/// GPU parameters for separable Gaussian blur.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GaussianParams {
    pub width:  u32,
    pub height: u32,
    pub sigma:  f32,
    pub radius: u32,
}


/// GPU parameters for image pyramid downsampling.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct PyramidParams {
    pub src_width:  u32,
    pub src_height: u32,
    pub dst_width:  u32,
    pub dst_height: u32,
}

/// GPU parameters for Sobel gradient computation.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SobelParams {
    pub width:  u32,
    pub height: u32,
}

/// GPU parameters for Canny hysteresis thresholding.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct CannyParams {
    pub width:          u32,
    pub height:         u32,
    pub low_threshold:  f32,
    pub high_threshold: f32,
}

/// GPU parameters for dense optical flow.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct FlowParams {
    pub width:  u32,
    pub height: u32,
    pub alpha:  f32,
}

/// GPU parameters for bilinear resize.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ResizeParams {
    pub src_width:  u32,
    pub src_height: u32,
    pub dst_width:  u32,
    pub dst_height: u32,
}

/// GPU parameters for affine warp (2x3 matrix).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct WarpAffineParams {
    pub width:      u32,
    pub height:     u32,
    pub src_width:  u32,
    pub src_height: u32,
    pub m00: f32, pub m01: f32, pub m02: f32,
    pub m10: f32, pub m11: f32, pub m12: f32,
}

/// GPU parameters for perspective warp (3x3 matrix).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct WarpPerspectiveParams {
    pub width:      u32,
    pub height:     u32,
    pub src_width:  u32,
    pub src_height: u32,
    pub m00: f32, pub m01: f32, pub m02: f32,
    pub m10: f32, pub m11: f32, pub m12: f32,
    pub m20: f32, pub m21: f32, pub m22: f32,
}

/// GPU parameters for integral image computation.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct IntegralParams {
    pub width:  u32,
    pub height: u32,
}

/// GPU parameters for color space conversion.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ColorParams {
    pub width:  u32,
    pub height: u32,
}

/// GPU parameters for morphological operations.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct MorphParams {
    pub width:    u32,
    pub height:   u32,
    pub radius_x: i32,
    pub radius_y: i32,
}

/// GPU parameters for binary thresholding.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ThresholdParams {
    pub width:     u32,
    pub height:    u32,
    pub threshold: f32,
    pub invert:    i32,
}

/// GPU parameters for adaptive thresholding.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct AdaptiveThresholdParams {
    pub width:  u32,
    pub height: u32,
    pub radius: i32,
    pub c:      f32,
    pub invert: i32,
}

/// GPU parameters for histogram computation.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct HistogramParams {
    pub width:  u32,
    pub height: u32,
}

/// GPU parameters for homography scoring.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct HomographyParams {
    pub n_points:         u32,
    pub inlier_threshold: f32,
    pub h00: f32, pub h01: f32, pub h02: f32,
    pub h10: f32, pub h11: f32, pub h12: f32,
    pub h20: f32, pub h21: f32, pub h22: f32,
}

/// Source-destination point correspondence for homography estimation.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct PointPair {
    pub src_x: f32,
    pub src_y: f32,
    pub dst_x: f32,
    pub dst_y: f32,
}

/// Per-point inlier classification from RANSAC scoring.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ScoreResult {
    pub error:     f32,
    pub is_inlier: u32,
}

/// GPU parameters for bilateral filtering.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct BilateralParams {
    pub width:         u32,
    pub height:        u32,
    pub radius:        i32,
    pub sigma_spatial: f32,
    pub sigma_range:   f32,
}

/// GPU parameters for Difference-of-Gaussians subtraction.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct DoGParams {
    pub width:  u32,
    pub height: u32,
}

/// GPU parameters for DoG extrema detection.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct DoGExtremaParams {
    pub width:              u32,
    pub height:             u32,
    pub contrast_threshold: f32,
    pub max_keypoints:      u32,
    pub octave:             u32,
    pub level:              u32,
}

/// Scale-space keypoint from DoG extrema detection.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct DoGKeypoint {
    pub position: [f32; 2],
    pub response: f32,
    pub octave:   u32,
    pub level:    u32,
    pub _pad0:    f32,
    pub _pad1:    f32,
    pub _pad2:    f32,
}

/// GPU parameters for brute-force descriptor matching.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct MatcherParams {
    pub n_query:      u32,
    pub n_train:      u32,
    pub max_hamming:  u32,
    pub ratio_thresh: f32,
}

/// Descriptor match with distance and Lowe's ratio.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct MatchResult {
    pub query_idx: u32,
    pub train_idx: u32,
    pub distance:  u32,
    pub ratio:     f32,
}

/// GPU parameters for NCC template matching.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct TemplateParams {
    pub img_width:  u32,
    pub img_height: u32,
    pub tpl_width:  u32,
    pub tpl_height: u32,
    pub tpl_mean:   f32,
    pub tpl_norm:   f32,
}

/// GPU parameters for Hough vote accumulation.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct HoughVoteParams {
    pub width:          u32,
    pub height:         u32,
    pub n_theta:        u32,
    pub n_rho:          u32,
    pub rho_max:        f32,
    pub edge_threshold: f32,
}

/// GPU parameters for Hough peak extraction.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct HoughPeakParams {
    pub n_theta:        u32,
    pub n_rho:          u32,
    pub vote_threshold: u32,
    pub max_lines:      u32,
    pub rho_max:        f32,
    pub nms_radius:     u32,
}

/// Detected line in polar coordinates (rho, theta) with vote count.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct HoughLine {
    pub rho:   f32,
    pub theta: f32,
    pub votes: u32,
    pub _pad:  u32,
}

/// GPU parameters for Jump Flooding distance transform.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct JFAParams {
    pub width:     u32,
    pub height:    u32,
    pub step_size: i32,
    pub threshold: f32,
}

/// GPU parameters for connected component labeling.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct CCLParams {
    pub width:     u32,
    pub height:    u32,
    pub threshold: f32,
}

/// GPU parameters for indirect dispatch argument setup.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct IndirectSetupParams {
    pub threads_per_threadgroup: u32,
}

/// Indirect dispatch arguments (mirrors `MTLDispatchThreadgroupsIndirectArguments`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct IndirectArgs {
    pub threadgroups_x: u32,
    pub threadgroups_y: u32,
    pub threadgroups_z: u32,
}