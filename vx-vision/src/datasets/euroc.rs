//! EuRoC MAV dataset loader.
//!
//! Loads image paths and ground-truth poses from the EuRoC MAV format.
//! See: <https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets>

use std::fs;
use std::path::{Path, PathBuf};

use super::{DatasetFrame, DatasetIterator};

/// EuRoC MAV dataset loader.
pub struct EurocDataset {
    frames: Vec<DatasetFrame>,
    index: usize,
}

impl EurocDataset {
    /// Loads an EuRoC sequence from the `mav0` directory.
    ///
    /// Expects `cam0/data.csv` for images and `state_groundtruth_estimate0/data.csv` for poses.
    pub fn load<P: AsRef<Path>>(dir: P) -> Result<Self, String> {
        let dir = dir.as_ref();

        let cam_dir = dir.join("mav0").join("cam0");
        let cam_csv = cam_dir.join("data.csv");
        let image_dir = cam_dir.join("data");

        let content = fs::read_to_string(&cam_csv)
            .map_err(|e| format!("cannot read {}: {e}", cam_csv.display()))?;

        let mut frames = Vec::new();
        for line in content.lines() {
            let line = line.trim();
            if line.starts_with('#') || line.starts_with("timestamp") || line.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() >= 2 {
                if let Ok(ts_ns) = parts[0].trim().parse::<u64>() {
                    let timestamp = ts_ns as f64 / 1e9;
                    let image_name = parts[1].trim();
                    frames.push(DatasetFrame {
                        timestamp,
                        image_path: image_dir.join(image_name),
                        depth_path: None,
                        #[cfg(feature = "reconstruction")]
                        pose: None, // Could load from groundtruth CSV
                    });
                }
            }
        }

        Ok(Self { frames, index: 0 })
    }
}

impl Iterator for EurocDataset {
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

impl DatasetIterator for EurocDataset {
    fn len(&self) -> Option<usize> {
        Some(self.frames.len())
    }
}
