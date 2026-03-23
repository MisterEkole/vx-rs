//! KITTI odometry dataset loader.
//!
//! Loads image paths and ground-truth poses from the KITTI odometry format.
//! See: <https://www.cvlibs.net/datasets/kitti/eval_odometry.php>

use std::fs;
use std::path::{Path, PathBuf};

use super::{DatasetFrame, DatasetIterator};

/// KITTI odometry dataset loader.
pub struct KittiDataset {
    frames: Vec<DatasetFrame>,
    index: usize,
}

impl KittiDataset {
    /// Loads a KITTI odometry sequence.
    ///
    /// `sequence_dir` should contain `image_0/` (or `image_2/`) and optionally `poses.txt`.
    pub fn load<P: AsRef<Path>>(
        sequence_dir: P,
        poses_file: Option<&Path>,
    ) -> Result<Self, String> {
        let dir = sequence_dir.as_ref();

        // Try image_2 (color) first, fall back to image_0 (gray)
        let image_dir = if dir.join("image_2").exists() {
            dir.join("image_2")
        } else {
            dir.join("image_0")
        };

        let mut image_files: Vec<PathBuf> = fs::read_dir(&image_dir)
            .map_err(|e| format!("cannot read {}: {e}", image_dir.display()))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "png"))
            .collect();
        image_files.sort();

        #[cfg(feature = "reconstruction")]
        let poses = poses_file.and_then(|p| Self::parse_poses(p).ok());

        let mut frames = Vec::new();
        for (i, path) in image_files.iter().enumerate() {
            #[cfg(feature = "reconstruction")]
            let pose = poses.as_ref().and_then(|p| p.get(i).copied());

            frames.push(DatasetFrame {
                timestamp: i as f64 / 10.0, // KITTI ~10 Hz
                image_path: path.clone(),
                depth_path: None,
                #[cfg(feature = "reconstruction")]
                pose,
            });
        }

        Ok(Self { frames, index: 0 })
    }

    #[cfg(feature = "reconstruction")]
    fn parse_poses(path: &Path) -> Result<Vec<crate::types_3d::CameraExtrinsics>, String> {
        let content =
            fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;

        let mut poses = Vec::new();
        for line in content.lines() {
            let vals: Vec<f32> = line
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();
            if vals.len() >= 12 {
                // KITTI format: 3x4 row-major [R|t]
                let rotation = [
                    vals[0], vals[1], vals[2], vals[4], vals[5], vals[6], vals[8], vals[9],
                    vals[10],
                ];
                let translation = [vals[3], vals[7], vals[11]];
                poses.push(crate::types_3d::CameraExtrinsics::new(
                    rotation,
                    translation,
                ));
            }
        }
        Ok(poses)
    }
}

impl Iterator for KittiDataset {
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

impl DatasetIterator for KittiDataset {
    fn len(&self) -> Option<usize> {
        Some(self.frames.len())
    }
}
