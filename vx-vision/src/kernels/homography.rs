//! RANSAC homography estimation with GPU-parallel scoring.

use core::ffi::c_void;
use core::ptr::NonNull;
use std::mem;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice,
    MTLLibrary, MTLSize,
};

use vx_gpu::UnifiedBuffer;
use crate::context::Context;
use crate::types::{HomographyParams, PointPair, ScoreResult};

/// RANSAC configuration for homography estimation.
#[derive(Clone, Debug)]
pub struct RansacConfig {
    /// Maximum RANSAC iterations.
    pub max_iterations: u32,

    /// Inlier reprojection error threshold in pixels.
    pub inlier_threshold: f32,

    /// Minimum inlier count to accept a model.
    pub min_inliers: u32,
}

impl Default for RansacConfig {
    fn default() -> Self {
        Self {
            max_iterations:   500,
            inlier_threshold: 3.0,
            min_inliers:      10,
        }
    }
}

/// Homography estimation result.
#[derive(Clone, Debug)]
pub struct HomographyResult {
    /// 3x3 homography matrix, row-major.
    pub homography: [f32; 9],

    /// Inlier count for the best model.
    pub n_inliers: u32,

    /// Per-pair inlier mask.
    pub inlier_mask: Vec<bool>,
}

/// RANSAC homography estimator with GPU scoring.
pub struct HomographyEstimator {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl HomographyEstimator {
    pub fn new(ctx: &Context) -> Result<Self, String> {
        let name = objc2_foundation::ns_string!("homography_score");
        let func = ctx.library().newFunctionWithName(name)
            .ok_or_else(|| "Missing kernel function 'homography_score'".to_string())?;
        let pipeline = ctx.device()
            .newComputePipelineStateWithFunction_error(&func)
            .map_err(|e| format!("Failed to create homography_score pipeline: {e}"))?;
        Ok(Self { pipeline })
    }

    /// Estimates a homography from at least 4 point correspondences.
    pub fn estimate(
        &self,
        ctx:    &Context,
        pairs:  &[PointPair],
        config: &RansacConfig,
    ) -> Result<HomographyResult, String> {
        let n = pairs.len();
        if n < 4 {
            return Err("Need at least 4 point pairs for homography estimation".into());
        }

        // Upload point pairs
        let mut pairs_buf: UnifiedBuffer<PointPair> = UnifiedBuffer::new(ctx.device(), n)?;
        pairs_buf.write(pairs);

        let results_buf: UnifiedBuffer<ScoreResult> = UnifiedBuffer::new(ctx.device(), n)?;
        let mut count_buf: UnifiedBuffer<u32> = UnifiedBuffer::new(ctx.device(), 1)?;

        let mut best_h = [0.0f32; 9];
        let mut best_inliers = 0u32;
        let mut best_mask = vec![false; n];

        // Deterministic LCG for sampling
        let mut rng_state: u64 = 0x12345678_9ABCDEF0;

        for _ in 0..config.max_iterations {
            // Sample 4 random correspondences
            let indices = random_4(&mut rng_state, n);

            // DLT solve
            let sample: Vec<PointPair> = indices.iter().map(|&i| pairs[i]).collect();
            let h = match dlt_homography(&sample) {
                Some(h) => h,
                None => continue,
            };

            // GPU scoring
            count_buf.write(&[0u32]);

            let params = HomographyParams {
                n_points:         n as u32,
                inlier_threshold: config.inlier_threshold,
                h00: h[0], h01: h[1], h02: h[2],
                h10: h[3], h11: h[4], h12: h[5],
                h20: h[6], h21: h[7], h22: h[8],
            };

            let _pairs_guard   = pairs_buf.gpu_guard();
            let _results_guard = results_buf.gpu_guard();
            let _count_guard   = count_buf.gpu_guard();

            let cmd_buf = ctx.queue().commandBuffer()
                .ok_or("Failed to create command buffer")?;
            let encoder = cmd_buf.computeCommandEncoder()
                .ok_or("Failed to create compute encoder")?;

            unsafe {
                encoder.setComputePipelineState(&self.pipeline);
                encoder.setBuffer_offset_atIndex(Some(pairs_buf.metal_buffer()),   0, 0);
                encoder.setBuffer_offset_atIndex(Some(results_buf.metal_buffer()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(count_buf.metal_buffer()),   0, 2);
                encoder.setBytes_length_atIndex(
                    NonNull::new_unchecked(&params as *const HomographyParams as *mut c_void),
                    mem::size_of::<HomographyParams>(),
                    3,
                );

                let tew     = self.pipeline.threadExecutionWidth();
                let grid    = MTLSize { width: n,   height: 1, depth: 1 };
                let tg_size = MTLSize { width: tew, height: 1, depth: 1 };
                encoder.dispatchThreads_threadsPerThreadgroup(grid, tg_size);
            }

            encoder.endEncoding();
            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();

            drop(_pairs_guard);
            drop(_results_guard);
            drop(_count_guard);

            let n_inliers = count_buf.as_slice()[0];
            if n_inliers > best_inliers {
                best_inliers = n_inliers;
                best_h = h;

                // Update inlier mask
                for (i, result) in results_buf.as_slice()[..n].iter().enumerate() {
                    best_mask[i] = result.is_inlier != 0;
                }

                // Early exit when inlier ratio is high
                if n_inliers as f32 > n as f32 * 0.9 {
                    break;
                }
            }
        }

        if best_inliers < config.min_inliers {
            return Err(format!(
                "RANSAC failed: best model has {} inliers, need at least {}",
                best_inliers, config.min_inliers
            ));
        }

        Ok(HomographyResult {
            homography: best_h,
            n_inliers: best_inliers,
            inlier_mask: best_mask,
        })
    }
}

/// Returns 4 unique indices in `[0, n)` via a simple LCG.
fn random_4(state: &mut u64, n: usize) -> [usize; 4] {
    let mut indices = [0usize; 4];
    let mut count = 0;
    while count < 4 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let idx = ((*state >> 33) as usize) % n;
        if !indices[..count].contains(&idx) {
            indices[count] = idx;
            count += 1;
        }
    }
    indices
}

/// Solves a 4-point DLT homography. Returns `None` if degenerate.
fn dlt_homography(pairs: &[PointPair]) -> Option<[f32; 9]> {
    assert!(pairs.len() >= 4);

    // Build the 8x9 constraint matrix
    let mut a = [[0.0f64; 9]; 8];

    for (i, p) in pairs.iter().take(4).enumerate() {
        let (x, y) = (p.src_x as f64, p.src_y as f64);
        let (xp, yp) = (p.dst_x as f64, p.dst_y as f64);

        a[2*i]   = [-x, -y, -1.0, 0.0, 0.0, 0.0, xp*x, xp*y, xp];
        a[2*i+1] = [0.0, 0.0, 0.0, -x, -y, -1.0, yp*x, yp*y, yp];
    }

    // Reduce to 8x8 by setting h[8] = 1
    let mut m = [[0.0f64; 8]; 8];
    let mut b = [0.0f64; 8];

    for i in 0..8 {
        for j in 0..8 {
            m[i][j] = a[i][j];
        }
        b[i] = -a[i][8];
    }

    // Solve via Gaussian elimination
    let h8 = gauss_solve_8x8(&mut m, &mut b)?;

    let mut h = [0.0f32; 9];
    for i in 0..8 {
        h[i] = h8[i] as f32;
    }
    h[8] = 1.0;

    // Normalize
    let norm: f32 = h.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-10 { return None; }
    for v in &mut h { *v /= norm; }

    Some(h)
}

/// Solves an 8x8 system via Gaussian elimination with partial pivoting.
fn gauss_solve_8x8(m: &mut [[f64; 8]; 8], b: &mut [f64; 8]) -> Option<[f64; 8]> {
    for col in 0..8 {
        // Partial pivoting
        let mut max_row = col;
        let mut max_val = m[col][col].abs();
        for row in (col + 1)..8 {
            if m[row][col].abs() > max_val {
                max_val = m[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-12 { return None; }

        // Row swap
        if max_row != col {
            m.swap(col, max_row);
            b.swap(col, max_row);
        }

        // Forward elimination
        let pivot = m[col][col];
        for row in (col + 1)..8 {
            let factor = m[row][col] / pivot;
            for j in col..8 {
                m[row][j] -= factor * m[col][j];
            }
            b[row] -= factor * b[col];
        }
    }

    // Back-substitution
    let mut x = [0.0f64; 8];
    for col in (0..8).rev() {
        let mut sum = b[col];
        for j in (col + 1)..8 {
            sum -= m[col][j] * x[j];
        }
        x[col] = sum / m[col][col];
    }

    Some(x)
}
