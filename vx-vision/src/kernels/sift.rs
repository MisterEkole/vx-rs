//! SIFT-like feature detection and description (GPU keypoints, CPU descriptors).

use crate::context::Context;
use crate::error::Result;
use crate::kernels::dog::{DoGConfig, DoGDetector};
use crate::kernels::gaussian::GaussianBlur;
use crate::kernels::pyramid::PyramidBuilder;
use crate::texture::Texture;
use crate::types::DoGKeypoint;

/// SIFT feature with position, scale, orientation, and 128-dim descriptor.
#[derive(Clone, Debug)]
pub struct SiftFeature {
    /// Position in the original image.
    pub x: f32,
    pub y: f32,
    /// Detection scale (sigma).
    pub scale: f32,
    /// Dominant orientation (radians).
    pub orientation: f32,
    /// 128-dim descriptor (4x4 spatial bins, 8 orientation bins).
    pub descriptor: [f32; 128],
}

/// SIFT pipeline configuration.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SiftConfig {
    /// Pyramid octaves. `0` selects automatically from image size.
    pub n_octaves: usize,
    /// Scale levels per octave.
    pub n_levels: usize,
    /// Base Gaussian sigma.
    pub base_sigma: f32,
    /// DoG contrast threshold.
    pub contrast_threshold: f32,
    /// Edge response threshold (principal curvature ratio).
    pub edge_threshold: f32,
    /// Maximum keypoint count.
    pub max_keypoints: u32,
    /// Descriptor patch radius in scale-relative pixels.
    pub descriptor_radius: f32,
}

impl Default for SiftConfig {
    fn default() -> Self {
        Self {
            n_octaves: 0, // auto
            n_levels: 3,
            base_sigma: 1.6,
            contrast_threshold: 0.04,
            edge_threshold: 10.0,
            max_keypoints: 2048,
            descriptor_radius: 6.0,
        }
    }
}

/// SIFT feature detection and description pipeline.
pub struct SiftPipeline {
    blur: GaussianBlur,
    dog: DoGDetector,
    pyramid: PyramidBuilder,
}

impl SiftPipeline {
    pub fn new(ctx: &Context) -> Result<Self> {
        let blur = GaussianBlur::new(ctx)?;
        let dog = DoGDetector::new(ctx)?;
        let pyramid = PyramidBuilder::new(ctx)?;
        Ok(Self { blur, dog, pyramid })
    }

    /// Detects and describes SIFT features in a grayscale image.
    pub fn detect_and_describe(
        &self,
        ctx: &Context,
        input: &Texture,
        config: &SiftConfig,
    ) -> Result<Vec<SiftFeature>> {
        let w = input.width();
        let h = input.height();

        let n_octaves = if config.n_octaves == 0 {
            let min_dim = w.min(h) as f64;
            (min_dim.log2() - 2.0).floor().max(1.0) as usize
        } else {
            config.n_octaves
        };

        let mut all_keypoints: Vec<(DoGKeypoint, usize)> = Vec::new();

        let pyramid_levels = self.pyramid.build(ctx, input, n_octaves.min(4))?;

        let mut octave_inputs: Vec<&Texture> = Vec::with_capacity(n_octaves);
        octave_inputs.push(input);
        for level in &pyramid_levels {
            octave_inputs.push(level);
        }

        for (oct, oct_input) in octave_inputs.iter().enumerate().take(n_octaves) {
            let dog_config = DoGConfig {
                n_levels: config.n_levels + 1,
                base_sigma: config.base_sigma,
                scale_factor: 2.0f32.powf(1.0 / config.n_levels as f32),
                contrast_threshold: config.contrast_threshold,
                max_keypoints: config.max_keypoints,
            };

            let keypoints = self.dog.detect(ctx, &self.blur, oct_input, &dog_config)?;

            for kp in keypoints {
                if kp.response.abs() >= config.contrast_threshold {
                    all_keypoints.push((kp, oct));
                }
            }
        }

        all_keypoints.sort_by(|a, b| b.0.response.abs().partial_cmp(&a.0.response.abs()).unwrap());
        all_keypoints.truncate(config.max_keypoints as usize);

        let img_data = input.read_gray8();

        let features: Vec<SiftFeature> = all_keypoints
            .iter()
            .filter_map(|(kp, octave)| {
                let scale_factor = (1 << octave) as f32;
                let x = kp.position[0] * scale_factor;
                let y = kp.position[1] * scale_factor;
                let sigma = config.base_sigma * 2.0f32.powi(*octave as i32);

                if x < 1.0 || y < 1.0 || x >= (w - 1) as f32 || y >= (h - 1) as f32 {
                    return None;
                }

                let orientation = compute_orientation(&img_data, w, h, x, y, sigma * 1.5);

                let descriptor = compute_descriptor(
                    &img_data,
                    w,
                    h,
                    x,
                    y,
                    sigma,
                    orientation,
                    config.descriptor_radius,
                );

                Some(SiftFeature {
                    x,
                    y,
                    scale: sigma,
                    orientation,
                    descriptor,
                })
            })
            .collect();

        Ok(features)
    }

    /// Detects keypoints without computing descriptors.
    pub fn detect(
        &self,
        ctx: &Context,
        input: &Texture,
        config: &SiftConfig,
    ) -> Result<Vec<DoGKeypoint>> {
        let dog_config = DoGConfig {
            n_levels: config.n_levels + 1,
            base_sigma: config.base_sigma,
            scale_factor: 2.0f32.powf(1.0 / config.n_levels as f32),
            contrast_threshold: config.contrast_threshold,
            max_keypoints: config.max_keypoints,
        };

        self.dog.detect(ctx, &self.blur, input, &dog_config)
    }
}

unsafe impl Send for SiftPipeline {}
unsafe impl Sync for SiftPipeline {}

impl SiftPipeline {
    /// Matches two feature sets using L2 distance with ratio test.
    pub fn match_features(
        query: &[SiftFeature],
        train: &[SiftFeature],
        ratio_thresh: f32,
    ) -> Vec<SiftMatch> {
        let mut matches = Vec::new();

        for (qi, q) in query.iter().enumerate() {
            let mut best_dist = f32::MAX;
            let mut second_dist = f32::MAX;
            let mut best_ti = 0;

            for (ti, t) in train.iter().enumerate() {
                let dist = l2_distance(&q.descriptor, &t.descriptor);
                if dist < best_dist {
                    second_dist = best_dist;
                    best_dist = dist;
                    best_ti = ti;
                } else if dist < second_dist {
                    second_dist = dist;
                }
            }

            if second_dist > 0.0 && best_dist / second_dist < ratio_thresh {
                matches.push(SiftMatch {
                    query_idx: qi,
                    train_idx: best_ti,
                    distance: best_dist,
                    ratio: best_dist / second_dist,
                });
            }
        }

        matches
    }
}

/// Match between two SIFT features.
#[derive(Clone, Debug)]
pub struct SiftMatch {
    pub query_idx: usize,
    pub train_idx: usize,
    pub distance: f32,
    pub ratio: f32,
}

fn compute_orientation(img: &[u8], w: u32, h: u32, x: f32, y: f32, sigma: f32) -> f32 {
    let mut hist = [0.0f32; 36];
    let radius = (sigma * 3.0).ceil() as i32;
    let xi = x as i32;
    let yi = y as i32;
    let inv_2s2 = 1.0 / (2.0 * sigma * sigma);

    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let px = xi + dx;
            let py = yi + dy;
            if px < 1 || py < 1 || px >= (w as i32 - 1) || py >= (h as i32 - 1) {
                continue;
            }

            let gx = img[(py as usize) * (w as usize) + (px as usize + 1)] as f32
                - img[(py as usize) * (w as usize) + (px as usize - 1)] as f32;
            let gy = img[((py + 1) as usize) * (w as usize) + px as usize] as f32
                - img[((py - 1) as usize) * (w as usize) + px as usize] as f32;

            let mag = (gx * gx + gy * gy).sqrt();
            let angle = gy.atan2(gx); // -pi to pi
            let weight = (-(dx * dx + dy * dy) as f32 * inv_2s2).exp();

            let deg = (angle.to_degrees() + 360.0) % 360.0;
            let bin = (deg / 10.0) as usize % 36;
            hist[bin] += mag * weight;
        }
    }

    let mut max_val = 0.0f32;
    let mut max_bin = 0;
    for (i, &v) in hist.iter().enumerate() {
        if v > max_val {
            max_val = v;
            max_bin = i;
        }
    }

    (max_bin as f32 * 10.0 + 5.0).to_radians()
}

#[allow(clippy::too_many_arguments)]
fn compute_descriptor(
    img: &[u8],
    w: u32,
    h: u32,
    x: f32,
    y: f32,
    sigma: f32,
    orientation: f32,
    radius: f32,
) -> [f32; 128] {
    let mut desc = [0.0f32; 128];
    let cos_t = orientation.cos();
    let sin_t = orientation.sin();
    let xi = x as i32;
    let yi = y as i32;
    let window = (radius * sigma).ceil() as i32;
    let bin_size = radius * sigma / 2.0;

    for dy in -window..=window {
        for dx in -window..=window {
            let px = xi + dx;
            let py = yi + dy;
            if px < 1 || py < 1 || px >= (w as i32 - 1) || py >= (h as i32 - 1) {
                continue;
            }

            let rx = (dx as f32) * cos_t + (dy as f32) * sin_t;
            let ry = -(dx as f32) * sin_t + (dy as f32) * cos_t;

            let bx = (rx / bin_size + 2.0) as i32;
            let by = (ry / bin_size + 2.0) as i32;

            if !(0..4).contains(&bx) || !(0..4).contains(&by) {
                continue;
            }

            let gx = img[(py as usize) * (w as usize) + (px as usize + 1)] as f32
                - img[(py as usize) * (w as usize) + (px as usize - 1)] as f32;
            let gy = img[((py + 1) as usize) * (w as usize) + px as usize] as f32
                - img[((py - 1) as usize) * (w as usize) + px as usize] as f32;

            let mag = (gx * gx + gy * gy).sqrt();
            let angle = gy.atan2(gx) - orientation;
            let deg = (angle.to_degrees() + 360.0) % 360.0;
            let obin = (deg / 45.0) as usize % 8;

            let inv_2s2 = 1.0 / (2.0 * (radius * sigma * 0.5).powi(2));
            let weight = (-(dx * dx + dy * dy) as f32 * inv_2s2).exp();

            let idx = (by as usize * 4 + bx as usize) * 8 + obin;
            if idx < 128 {
                desc[idx] += mag * weight;
            }
        }
    }

    let mut norm = 0.0f32;
    for v in &desc {
        norm += v * v;
    }
    norm = norm.sqrt().max(1e-7);
    for v in &mut desc {
        *v /= norm;
    }

    for v in &mut desc {
        if *v > 0.2 {
            *v = 0.2;
        }
    }
    let mut norm2 = 0.0f32;
    for v in &desc {
        norm2 += v * v;
    }
    norm2 = norm2.sqrt().max(1e-7);
    for v in &mut desc {
        *v /= norm2;
    }

    desc
}

fn l2_distance(a: &[f32; 128], b: &[f32; 128]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..128 {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum.sqrt()
}
