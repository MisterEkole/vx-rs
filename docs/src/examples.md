# Examples Walkthrough

All examples take an image path as argument:

```bash
cargo run --release --example <name> -- path/to/image.png
```

## fast_demo

Detects FAST corners, scores with Harris, suppresses with NMS. Prints corner count and timing.

**Demonstrates:** Feature detection pipeline, `FastDetector`, `HarrisScorer`, `NmsSuppressor`.

## edge_detection_demo

Runs Gaussian blur → Sobel gradients → Canny edges. Reports timing for each stage and pixel statistics.

**Demonstrates:** Image processing chain, `GaussianBlur`, `SobelFilter`, `CannyDetector`.

## threshold_demo

Compares thresholding methods: histogram analysis, Otsu's automatic threshold, fixed binary, integral image computation, and adaptive threshold. Prints timing and foreground percentages.

**Demonstrates:** `Histogram`, `Threshold` (all modes), `IntegralImage`.

## advanced_cv_demo

Runs five algorithms on one image: bilateral filter, Canny + Hough line detection, Otsu + distance transform, connected components, and template matching (self-patch). Prints detailed results for each.

**Demonstrates:** Full range of analysis kernels.

## feature_matching_demo

Detects ORB features in two images, matches with brute-force Hamming distance, and reports match statistics. Also runs SIFT detection for comparison.

**Demonstrates:** `OrbDescriptor`, `BruteMatcher`, `SiftPipeline`.

## klt_benchmark

Loads a sequence of PNG frames (e.g., from EuRoC dataset), detects FAST corners on the first frame, then tracks them through subsequent frames using KLT optical flow. Reports per-frame timing and track survival rate.

**Demonstrates:** `KltTracker`, multi-frame processing, re-detection strategy.

## orb_stereo_benchmark

Runs the full stereo pipeline on synthetic or real stereo pairs: FAST detection, Harris scoring, NMS, ORB descriptors, stereo matching with epipolar constraints. Reports 3D point triangulation results.

**Demonstrates:** `StereoMatcher`, full detection-to-3D pipeline.

## pipeline_pool_demo

Benchmarks three approaches to multi-frame processing: individual dispatches, pipeline batching, and pipeline + TexturePool. Reports timing comparison and pool hit rates.

**Demonstrates:** `Pipeline`, `TexturePool`, performance comparison.
