// examples/point_cloud_processing_demo.rs
//
// Synthetic sphere -> normal estimation -> outlier removal -> voxel downsample -> PLY export.
//
// Run with:
//   cargo run --release --example point_cloud_processing_demo --features reconstruction

use std::time::Instant;

use vx_vision::kernels::normal_estimation::{NormalEstimator, NormalEstimatorConfig};
use vx_vision::kernels::outlier_filter::{OutlierFilter, OutlierFilterConfig};
use vx_vision::kernels::voxel_downsample::{VoxelDownsample, VoxelDownsampleConfig};
use vx_vision::types_3d::{Point3D, PointCloud};
use vx_vision::Context;

fn main() {
    let ctx = Context::new().expect("No Metal GPU available");
    println!("VX Point Cloud Processing Demo");
    println!("===============================\n");

    // Generate synthetic sphere point cloud with noise
    let n_points = 10_000;
    let radius = 1.0f32;
    let mut points = Vec::with_capacity(n_points + 50);

    for i in 0..n_points {
        let phi = std::f32::consts::PI * (i as f32 / n_points as f32);
        let theta = 2.0 * std::f32::consts::PI * (i as f32 * 1.618033); // golden ratio spiral

        let noise = 0.02 * ((i * 7 + 3) % 100) as f32 / 100.0 - 0.01;
        let r = radius + noise;

        points.push(Point3D {
            position: [
                r * phi.sin() * theta.cos(),
                r * phi.sin() * theta.sin(),
                r * phi.cos(),
            ],
            color: [
                (128.0 + 127.0 * phi.sin() * theta.cos()) as u8,
                (128.0 + 127.0 * phi.sin() * theta.sin()) as u8,
                (128.0 + 127.0 * phi.cos()) as u8,
                255,
            ],
            normal: [0.0; 3],
        });
    }

    // Add outliers
    for i in 0..50 {
        points.push(Point3D {
            position: [5.0 + i as f32 * 0.1, 5.0 + i as f32 * 0.1, 5.0],
            color: [255, 0, 0, 255],
            ..Default::default()
        });
    }

    let cloud = PointCloud { points };
    println!("Generated sphere: {} points (+ 50 outliers)", n_points);

    // Estimate normals
    let estimator = NormalEstimator::new(&ctx).unwrap();
    let mut normal_config = NormalEstimatorConfig::default();
    normal_config.radius = 0.15;

    let t0 = Instant::now();
    let normals = estimator.compute(&ctx, &cloud, &normal_config).unwrap();
    println!(
        "Normal estimation: {:.1}ms",
        t0.elapsed().as_secs_f64() * 1000.0
    );

    // Apply normals to the cloud
    let mut cloud_with_normals = cloud.clone();
    for (p, n) in cloud_with_normals.points.iter_mut().zip(normals.iter()) {
        p.normal = *n;
    }

    let nonzero_normals = normals
        .iter()
        .filter(|n| n[0] != 0.0 || n[1] != 0.0 || n[2] != 0.0)
        .count();
    println!("  {} points with valid normals", nonzero_normals);

    // Remove outliers
    let filter = OutlierFilter::new(&ctx).unwrap();
    let filter_config = OutlierFilterConfig::default();

    let t1 = Instant::now();
    let filtered = filter
        .filter(&ctx, &cloud_with_normals, &filter_config)
        .unwrap();
    println!(
        "Outlier removal: {:.1}ms → {} → {} points",
        t1.elapsed().as_secs_f64() * 1000.0,
        cloud_with_normals.len(),
        filtered.len()
    );

    // Voxel downsampling
    let downsample = VoxelDownsample::new(&ctx).unwrap();
    let ds_config = VoxelDownsampleConfig::new(0.05);

    let t2 = Instant::now();
    let downsampled = downsample.downsample(&ctx, &filtered, &ds_config).unwrap();
    println!(
        "Voxel downsample: {:.1}ms → {} → {} points",
        t2.elapsed().as_secs_f64() * 1000.0,
        filtered.len(),
        downsampled.len()
    );

    // Export results
    vx_vision::export::write_ply_ascii("sphere_processed.ply", &downsampled).unwrap();
    println!("\nExported processed cloud to sphere_processed.ply");

    if let Some((min, max)) = downsampled.bounds() {
        println!(
            "  Bounds: [{:.2}, {:.2}, {:.2}] → [{:.2}, {:.2}, {:.2}]",
            min[0], min[1], min[2], max[0], max[1], max[2]
        );
    }

    println!(
        "\nTotal pipeline: {:.1}ms",
        t0.elapsed().as_secs_f64() * 1000.0
    );
}
