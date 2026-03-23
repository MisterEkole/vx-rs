//! TUM RGB-D dataset loader.
//!
//! Loads image paths and ground-truth poses from the TUM RGB-D benchmark format.
//! See: <https://vision.in.tum.de/data/datasets/rgbd-dataset>

use std::fs;
use std::path::{Path, PathBuf};

use super::{DatasetFrame, DatasetIterator};

/// TUM RGB-D dataset loader.
pub struct TumDataset {
    frames: Vec<DatasetFrame>,
    index: usize,
}

impl TumDataset {
    /// Loads a TUM RGB-D sequence from the given directory.
    ///
    /// Expects `rgb.txt`, `depth.txt`, and optionally `groundtruth.txt`.
    pub fn load<P: AsRef<Path>>(dir: P) -> Result<Self, String> {
        let dir = dir.as_ref();

        let rgb_list = Self::parse_file_list(&dir.join("rgb.txt"))?;
        let depth_list = Self::parse_file_list(&dir.join("depth.txt")).ok();

        #[cfg(feature = "reconstruction")]
        let poses = Self::parse_groundtruth(&dir.join("groundtruth.txt")).ok();

        let mut frames = Vec::new();
        for (timestamp, rgb_path) in &rgb_list {
            let image_path = dir.join(rgb_path);
            let depth_path = depth_list.as_ref().and_then(|dl| {
                // Find closest depth frame by timestamp
                dl.iter()
                    .min_by(|a, b| {
                        (a.0 - timestamp)
                            .abs()
                            .partial_cmp(&(b.0 - timestamp).abs())
                            .unwrap()
                    })
                    .filter(|d| (d.0 - timestamp).abs() < 0.05) // 50ms tolerance
                    .map(|d| dir.join(&d.1))
            });

            #[cfg(feature = "reconstruction")]
            let pose = poses.as_ref().and_then(|p| {
                p.iter()
                    .min_by(|a, b| {
                        (a.0 - timestamp)
                            .abs()
                            .partial_cmp(&(b.0 - timestamp).abs())
                            .unwrap()
                    })
                    .filter(|p| (p.0 - timestamp).abs() < 0.05)
                    .map(|p| p.1)
            });

            frames.push(DatasetFrame {
                timestamp: *timestamp,
                image_path,
                depth_path,
                #[cfg(feature = "reconstruction")]
                pose,
            });
        }

        Ok(Self { frames, index: 0 })
    }

    fn parse_file_list(path: &Path) -> Result<Vec<(f64, String)>, String> {
        let content =
            fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;

        let mut entries = Vec::new();
        for line in content.lines() {
            let line = line.trim();
            if line.starts_with('#') || line.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                if let Ok(ts) = parts[0].parse::<f64>() {
                    entries.push((ts, parts[1].to_string()));
                }
            }
        }
        Ok(entries)
    }

    #[cfg(feature = "reconstruction")]
    fn parse_groundtruth(
        path: &Path,
    ) -> Result<Vec<(f64, crate::types_3d::CameraExtrinsics)>, String> {
        let content =
            fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;

        let mut poses = Vec::new();
        for line in content.lines() {
            let line = line.trim();
            if line.starts_with('#') || line.is_empty() {
                continue;
            }
            let parts: Vec<f64> = line
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();
            if parts.len() >= 8 {
                // TUM format: timestamp tx ty tz qx qy qz qw
                let ts = parts[0];
                let tx = parts[1] as f32;
                let ty = parts[2] as f32;
                let tz = parts[3] as f32;
                let qx = parts[4] as f32;
                let qy = parts[5] as f32;
                let qz = parts[6] as f32;
                let qw = parts[7] as f32;

                // Quaternion to rotation matrix (row-major)
                let rotation = quat_to_rotation(qx, qy, qz, qw);
                let ext = crate::types_3d::CameraExtrinsics::new(rotation, [tx, ty, tz]);
                poses.push((ts, ext));
            }
        }
        Ok(poses)
    }
}

#[cfg(feature = "reconstruction")]
fn quat_to_rotation(qx: f32, qy: f32, qz: f32, qw: f32) -> [f32; 9] {
    let x2 = qx + qx;
    let y2 = qy + qy;
    let z2 = qz + qz;
    let xx = qx * x2;
    let xy = qx * y2;
    let xz = qx * z2;
    let yy = qy * y2;
    let yz = qy * z2;
    let zz = qz * z2;
    let wx = qw * x2;
    let wy = qw * y2;
    let wz = qw * z2;

    [
        1.0 - (yy + zz),
        xy - wz,
        xz + wy,
        xy + wz,
        1.0 - (xx + zz),
        yz - wx,
        xz - wy,
        yz + wx,
        1.0 - (xx + yy),
    ]
}

impl Iterator for TumDataset {
    type Item = DatasetFrame;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.frames.len() {
            let frame = self.frames[self.index].clone();
            self.index += 1;
            Some(frame)
        } else {
            None
        }
    }
}

impl DatasetIterator for TumDataset {
    fn len(&self) -> Option<usize> {
        Some(self.frames.len())
    }
}
