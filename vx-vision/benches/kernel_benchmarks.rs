//! Criterion benchmarks for core GPU kernels.
//!
//! Run with: cargo bench -p vx-vision

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use vx_vision::kernels::fast::{FastDetectConfig, FastDetector};
use vx_vision::kernels::gaussian::{GaussianBlur, GaussianConfig};
use vx_vision::kernels::harris::{HarrisConfig, HarrisScorer};
use vx_vision::kernels::nms::{NmsConfig, NmsSuppressor};
use vx_vision::kernels::orb::{OrbConfig, OrbDescriptor};
use vx_vision::Context;

/// Generates a synthetic gradient + noise image for benchmarking.
fn make_bench_image(w: u32, h: u32) -> Vec<u8> {
    (0..(w * h))
        .map(|i| {
            let x = i % w;
            let y = i / w;
            let grad = ((x as f32 / w as f32) * 255.0) as u8;
            let noise = ((x.wrapping_mul(17) ^ y.wrapping_mul(31)) % 64) as u8;
            grad.wrapping_add(noise)
        })
        .collect()
}

/// Standard ORB test pattern (256 pairs x 4 offsets = 1024 values).
fn orb_pattern() -> Vec<i32> {
    let mut pattern = Vec::with_capacity(1024);
    for i in 0..256u32 {
        let angle = (i as f32) * std::f32::consts::TAU / 256.0;
        let r1 = 5.0 + (i as f32 % 8.0);
        let r2 = 5.0 + ((i + 128) as f32 % 8.0);
        pattern.push((r1 * angle.cos()) as i32);
        pattern.push((r1 * angle.sin()) as i32);
        pattern.push((r2 * (angle + 0.5).cos()) as i32);
        pattern.push((r2 * (angle + 0.5).sin()) as i32);
    }
    pattern
}

fn bench_fast(c: &mut Criterion) {
    let ctx = Context::new().unwrap();
    let det = FastDetector::new(&ctx).unwrap();
    let cfg = FastDetectConfig::default();

    let mut group = c.benchmark_group("FAST");
    for &(w, h) in &[(752, 480), (1920, 1080)] {
        let pixels = make_bench_image(w, h);
        let texture = ctx.texture_gray8(&pixels, w, h).unwrap();

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{w}x{h}")),
            &(),
            |b, _| {
                b.iter(|| det.detect(&ctx, &texture, &cfg).unwrap());
            },
        );
    }
    group.finish();
}

fn bench_fast_harris_nms_orb(c: &mut Criterion) {
    let ctx = Context::new().unwrap();
    let fast = FastDetector::new(&ctx).unwrap();
    let harris = HarrisScorer::new(&ctx).unwrap();
    let nms = NmsSuppressor::new(&ctx).unwrap();
    let orb = OrbDescriptor::new(&ctx).unwrap();

    let fast_cfg = FastDetectConfig::default();
    let harris_cfg = HarrisConfig::default();
    let nms_cfg = NmsConfig::default();
    let orb_cfg = OrbConfig::default();
    let pattern = orb_pattern();

    let mut group = c.benchmark_group("FAST+Harris+NMS+ORB");
    for &(w, h) in &[(752, 480), (1920, 1080)] {
        let pixels = make_bench_image(w, h);
        let texture = ctx.texture_gray8(&pixels, w, h).unwrap();

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{w}x{h}")),
            &(),
            |b, _| {
                b.iter(|| {
                    let fast_result = fast.detect(&ctx, &texture, &fast_cfg).unwrap();
                    let scored = harris
                        .compute(&ctx, &texture, &fast_result.corners, &harris_cfg)
                        .unwrap();
                    let suppressed = nms.run(&ctx, &scored, &nms_cfg).unwrap();
                    let _orb_result = orb
                        .compute(&ctx, &texture, &suppressed, &pattern, &orb_cfg)
                        .unwrap();
                });
            },
        );
    }
    group.finish();
}

fn bench_pipeline_vs_individual(c: &mut Criterion) {
    let ctx = Context::new().unwrap();
    let gauss = GaussianBlur::new(&ctx).unwrap();

    let w = 1920u32;
    let h = 1080u32;
    let pixels = make_bench_image(w, h);
    let input = ctx.texture_gray8(&pixels, w, h).unwrap();

    let mut gauss_cfg = GaussianConfig::default();
    gauss_cfg.sigma = 1.5;
    gauss_cfg.radius = 4;

    let mut group = c.benchmark_group("Pipeline vs Individual");

    // Individual: 3 separate command buffers
    group.bench_function("3x Gaussian (individual)", |b| {
        let inter1 = ctx.texture_intermediate_gray8(w, h).unwrap();
        let inter2 = ctx.texture_intermediate_gray8(w, h).unwrap();
        let output = ctx.texture_output_gray8(w, h).unwrap();
        b.iter(|| {
            gauss.apply(&ctx, &input, &inter1, &gauss_cfg).unwrap();
            gauss.apply(&ctx, &inter1, &inter2, &gauss_cfg).unwrap();
            gauss.apply(&ctx, &inter2, &output, &gauss_cfg).unwrap();
        });
    });

    // Pipeline: single command buffer
    group.bench_function("3x Gaussian (pipeline)", |b| {
        let inter1 = ctx.texture_intermediate_gray8(w, h).unwrap();
        let inter2 = ctx.texture_intermediate_gray8(w, h).unwrap();
        let output = ctx.texture_output_gray8(w, h).unwrap();
        b.iter(|| {
            let pipe = vx_vision::Pipeline::begin(&ctx).unwrap();
            let cmd = pipe.cmd_buf();
            let s1 = gauss
                .encode(&ctx, cmd, &input, &inter1, &gauss_cfg)
                .unwrap();
            let s2 = gauss
                .encode(&ctx, cmd, &inter1, &inter2, &gauss_cfg)
                .unwrap();
            let s3 = gauss
                .encode(&ctx, cmd, &inter2, &output, &gauss_cfg)
                .unwrap();
            let _ = (s1, s2, s3);
            let _retained = pipe.commit_and_wait();
        });
    });

    group.finish();
}

// ── 3D Reconstruction benchmarks ──

#[cfg(feature = "reconstruction")]
fn make_bench_depth(w: u32, h: u32) -> Vec<f32> {
    (0..(w * h))
        .map(|i| {
            let x = (i % w) as f32 / w as f32;
            let y = (i / w) as f32 / h as f32;
            let cx = x - 0.5;
            let cy = y - 0.5;
            let r2 = cx * cx + cy * cy;
            if r2 < 0.2 {
                2.0 + r2 * 3.0
            } else {
                0.0
            }
        })
        .collect()
}

#[cfg(feature = "reconstruction")]
fn make_bench_cloud(n: usize) -> vx_vision::types_3d::PointCloud {
    use vx_vision::types_3d::{Point3D, PointCloud};
    let points: Vec<Point3D> = (0..n)
        .map(|i| {
            let t = i as f32 / n as f32;
            let phi = std::f32::consts::PI * t;
            let theta = std::f32::consts::TAU * t * 1.618;
            Point3D {
                position: [phi.sin() * theta.cos(), phi.sin() * theta.sin(), phi.cos()],
                ..Default::default()
            }
        })
        .collect();
    PointCloud { points }
}

#[cfg(feature = "reconstruction")]
fn bench_sgm_stereo(c: &mut Criterion) {
    use vx_vision::kernels::sgm::{SGMStereo, SGMStereoConfig};

    let ctx = Context::new().unwrap();
    let sgm = SGMStereo::new(&ctx).unwrap();
    let cfg = SGMStereoConfig::new(64);

    let mut group = c.benchmark_group("SGM Stereo");
    for &(w, h) in &[(320, 240), (640, 480)] {
        let left_px = make_bench_image(w, h);
        let right_px: Vec<u8> = left_px
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                if i % w as usize >= 8 {
                    left_px[i - 8]
                } else {
                    v
                }
            })
            .collect();
        let left = ctx.texture_gray8(&left_px, w, h).unwrap();
        let right = ctx.texture_gray8(&right_px, w, h).unwrap();
        let output = ctx.texture_output_r32float(w, h).unwrap();

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{w}x{h}")),
            &(),
            |b, _| {
                b.iter(|| {
                    sgm.compute_disparity(&ctx, &left, &right, &output, &cfg)
                        .unwrap()
                });
            },
        );
    }
    group.finish();
}

#[cfg(feature = "reconstruction")]
fn bench_depth_filter(c: &mut Criterion) {
    use vx_vision::kernels::depth_filter::{DepthFilter, DepthFilterConfig, DepthMedianConfig};

    let ctx = Context::new().unwrap();
    let filter = DepthFilter::new(&ctx).unwrap();

    let w = 640u32;
    let h = 480u32;
    let depth = make_bench_depth(w, h);
    let input = ctx.texture_r32float(&depth, w, h).unwrap();
    let output = ctx.texture_output_r32float(w, h).unwrap();

    let mut group = c.benchmark_group("Depth Filter");

    let bil_cfg = DepthFilterConfig::new(3, 5.0, 0.05);
    group.bench_function("bilateral 640x480", |b| {
        b.iter(|| {
            filter
                .apply_bilateral(&ctx, &input, &output, &bil_cfg)
                .unwrap()
        });
    });

    let med_cfg = DepthMedianConfig::default();
    group.bench_function("median 640x480", |b| {
        b.iter(|| {
            filter
                .apply_median(&ctx, &input, &output, &med_cfg)
                .unwrap()
        });
    });

    group.finish();
}

#[cfg(feature = "reconstruction")]
fn bench_depth_to_cloud(c: &mut Criterion) {
    use vx_vision::kernels::depth_to_cloud::{DepthToCloud, DepthToCloudConfig};
    use vx_vision::types_3d::{CameraIntrinsics, DepthMap};

    let ctx = Context::new().unwrap();
    let d2c = DepthToCloud::new(&ctx).unwrap();
    let cfg = DepthToCloudConfig::new(0.1, 10.0);

    let mut group = c.benchmark_group("Depth-to-Cloud");
    for &(w, h) in &[(640, 480), (1280, 720)] {
        let depth = make_bench_depth(w, h);
        let tex = ctx.texture_r32float(&depth, w, h).unwrap();
        let intrinsics = CameraIntrinsics::from_focal_length(w as f32, w, h);
        let dm = DepthMap::new(tex, intrinsics, 0.1, 10.0).unwrap();

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{w}x{h}")),
            &(),
            |b, _| {
                b.iter(|| d2c.compute(&ctx, &dm, None, &cfg).unwrap());
            },
        );
    }
    group.finish();
}

#[cfg(feature = "reconstruction")]
fn bench_depth_colorize(c: &mut Criterion) {
    use vx_vision::kernels::depth_colorize::{DepthColorize, DepthColorizeConfig};

    let ctx = Context::new().unwrap();
    let colorize = DepthColorize::new(&ctx).unwrap();
    let cfg = DepthColorizeConfig::new(0.5, 5.0);

    let w = 1920u32;
    let h = 1080u32;
    let depth = make_bench_depth(w, h);
    let input = ctx.texture_r32float(&depth, w, h).unwrap();
    let output = ctx.texture_output_rgba8(w, h).unwrap();

    c.bench_function("Depth Colorize 1920x1080", |b| {
        b.iter(|| colorize.apply(&ctx, &input, &output, &cfg).unwrap());
    });
}

#[cfg(feature = "reconstruction")]
fn bench_normal_estimation(c: &mut Criterion) {
    use vx_vision::kernels::normal_estimation::{NormalEstimator, NormalEstimatorConfig};

    let ctx = Context::new().unwrap();
    let estimator = NormalEstimator::new(&ctx).unwrap();

    let mut group = c.benchmark_group("Normal Estimation");

    // Organized (depth map)
    let w = 640u32;
    let h = 480u32;
    let depth = make_bench_depth(w, h);
    let input = ctx.texture_r32float(&depth, w, h).unwrap();
    let output = ctx.texture_output_rgba8(w, h).unwrap();
    group.bench_function("organized 640x480", |b| {
        b.iter(|| {
            estimator
                .compute_from_depth(&ctx, &input, &output, 500.0, 500.0, 320.0, 240.0)
                .unwrap()
        });
    });

    // Unorganized (point cloud)
    let cloud = make_bench_cloud(5000);
    let mut cfg = NormalEstimatorConfig::default();
    cfg.radius = 0.15;
    group.bench_function("unorganized 5K pts", |b| {
        b.iter(|| estimator.compute(&ctx, &cloud, &cfg).unwrap());
    });

    group.finish();
}

#[cfg(feature = "reconstruction")]
fn bench_outlier_filter(c: &mut Criterion) {
    use vx_vision::kernels::outlier_filter::{OutlierFilter, OutlierFilterConfig};

    let ctx = Context::new().unwrap();
    let filter = OutlierFilter::new(&ctx).unwrap();
    let cfg = OutlierFilterConfig::default();
    let cloud = make_bench_cloud(10_000);

    c.bench_function("Outlier Filter 10K pts", |b| {
        b.iter(|| filter.filter(&ctx, &cloud, &cfg).unwrap());
    });
}

#[cfg(feature = "reconstruction")]
fn bench_voxel_downsample(c: &mut Criterion) {
    use vx_vision::kernels::voxel_downsample::{VoxelDownsample, VoxelDownsampleConfig};

    let ctx = Context::new().unwrap();
    let ds = VoxelDownsample::new(&ctx).unwrap();
    let cfg = VoxelDownsampleConfig::new(0.1);
    let cloud = make_bench_cloud(50_000);

    c.bench_function("Voxel Downsample 50K pts", |b| {
        b.iter(|| ds.downsample(&ctx, &cloud, &cfg).unwrap());
    });
}

#[cfg(feature = "reconstruction")]
fn bench_tsdf_integrate(c: &mut Criterion) {
    use vx_vision::kernels::tsdf::{TSDFConfig, TSDFVolume};
    use vx_vision::types_3d::{CameraExtrinsics, CameraIntrinsics, DepthMap};

    let ctx = Context::new().unwrap();
    let config = TSDFConfig::default(); // 128³
    let tsdf = TSDFVolume::new(&ctx, config).unwrap();

    let w = 64u32;
    let h = 64u32;
    let intrinsics = CameraIntrinsics::new(200.0, 200.0, 32.0, 32.0, w, h);
    let depth = vec![0.32f32; (w * h) as usize];
    let tex = ctx.texture_r32float(&depth, w, h).unwrap();
    let dm = DepthMap::new(tex, intrinsics, 0.01, 2.0).unwrap();
    let pose = CameraExtrinsics::identity();

    c.bench_function("TSDF Integrate 128³", |b| {
        b.iter(|| tsdf.integrate(&ctx, &dm, &pose).unwrap());
    });
}

#[cfg(feature = "reconstruction")]
fn bench_marching_cubes(c: &mut Criterion) {
    use vx_vision::kernels::marching_cubes::{MarchingCubes, MarchingCubesConfig};

    let ctx = Context::new().unwrap();
    let res = [64u32, 64, 64];
    let vs = 0.05f32;
    let origin = [-1.6f32, -1.6, -1.6];
    let mut grid = vx_vision::types_3d::VoxelGrid::from_context(&ctx, res, vs, origin).unwrap();

    // Sphere SDF
    let total = 64 * 64 * 64;
    let mut sdf = vec![1.0f32; total];
    for iz in 0..64usize {
        for iy in 0..64usize {
            for ix in 0..64usize {
                let x = origin[0] + (ix as f32 + 0.5) * vs;
                let y = origin[1] + (iy as f32 + 0.5) * vs;
                let z = origin[2] + (iz as f32 + 0.5) * vs;
                sdf[(iz * 64 + iy) * 64 + ix] = (x * x + y * y + z * z).sqrt() - 1.0;
            }
        }
    }
    grid.tsdf.write(&sdf);
    grid.weights.write(&vec![1.0f32; total]);

    let mc_cfg = MarchingCubesConfig::default();
    c.bench_function("Marching Cubes 64³ sphere", |b| {
        b.iter(|| MarchingCubes::extract(&grid, &mc_cfg));
    });
}

criterion_group!(
    benches,
    bench_fast,
    bench_fast_harris_nms_orb,
    bench_pipeline_vs_individual,
);

#[cfg(feature = "reconstruction")]
criterion_group!(
    recon_benches,
    bench_sgm_stereo,
    bench_depth_filter,
    bench_depth_to_cloud,
    bench_depth_colorize,
    bench_normal_estimation,
    bench_outlier_filter,
    bench_voxel_downsample,
    bench_tsdf_integrate,
    bench_marching_cubes,
);

#[cfg(feature = "reconstruction")]
criterion_main!(benches, recon_benches);

#[cfg(not(feature = "reconstruction"))]
criterion_main!(benches);
