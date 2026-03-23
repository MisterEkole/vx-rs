// examples/depth_to_cloud_demo.rs
//
// Stereo matching -> depth colorization -> 3D point cloud -> PLY export.
//
// Run with:
//   cargo run --release --example depth_to_cloud_demo --features reconstruction -- path/to/image.png

use std::time::Instant;

use vx_vision::kernels::depth_colorize::{DepthColorize, DepthColorizeConfig};
use vx_vision::kernels::depth_to_cloud::{DepthToCloud, DepthToCloudConfig};
use vx_vision::kernels::sgm::{SGMStereo, SGMStereoConfig};
use vx_vision::types_3d::{CameraIntrinsics, DepthMap};
use vx_vision::Context;

fn main() {
    let ctx = Context::new().expect("No Metal GPU available");
    println!("VX Depth-to-Cloud Pipeline Demo");
    println!("================================\n");

    let path = std::env::args()
        .nth(1)
        .expect("Usage: depth_to_cloud_demo <image_path>");
    let img = image::open(&path).expect("Failed to open image").to_luma8();
    let (w, h) = img.dimensions();
    println!("Loaded {}×{} image: {}", w, h, path);

    // Create synthetic stereo pair by shifting the image
    let shift = 10u32;
    let left_data = img.as_raw();
    let right_data: Vec<u8> = (0..(w * h))
        .map(|i| {
            let x = i % w;
            if x >= shift {
                left_data[(i - shift) as usize]
            } else {
                0
            }
        })
        .collect();

    let left_tex = ctx.texture_gray8(left_data, w, h).unwrap();
    let right_tex = ctx.texture_gray8(&right_data, w, h).unwrap();

    // SGM stereo matching
    let sgm = SGMStereo::new(&ctx).unwrap();
    let output_disp = ctx.texture_output_r32float(w, h).unwrap();

    let t0 = Instant::now();
    let config = SGMStereoConfig::new(64);
    sgm.compute_disparity(&ctx, &left_tex, &right_tex, &output_disp, &config)
        .unwrap();
    println!("SGM stereo: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);

    let disp_data = output_disp.read_r32float();
    let nonzero_disp = disp_data.iter().filter(|&&d| d > 0.0).count();
    let max_disp = disp_data.iter().cloned().fold(0.0f32, f32::max);
    println!(
        "  {} pixels with valid disparity (max: {:.1})",
        nonzero_disp, max_disp
    );

    // Depth colorization
    let colorize = DepthColorize::new(&ctx).unwrap();
    let color_out = ctx.texture_output_rgba8(w, h).unwrap();
    let color_config = DepthColorizeConfig::new(0.0, max_disp.max(1.0));

    let t1 = Instant::now();
    colorize
        .apply(&ctx, &output_disp, &color_out, &color_config)
        .unwrap();
    println!(
        "Depth colorize: {:.1}ms",
        t1.elapsed().as_secs_f64() * 1000.0
    );

    // Depth-to-point-cloud
    let d2c = DepthToCloud::new(&ctx).unwrap();
    let intrinsics = CameraIntrinsics::from_focal_length(w as f32, w, h);

    // Create depth map from disparity (use disparity directly as "depth" for demo)
    let depth_tex = ctx.texture_r32float(&disp_data, w, h).unwrap();
    let depth_map = DepthMap::new(depth_tex, intrinsics, 1.0, max_disp.max(1.0)).unwrap();

    let d2c_config = DepthToCloudConfig::new(1.0, max_disp.max(1.0));

    let t2 = Instant::now();
    let cloud = d2c.compute(&ctx, &depth_map, None, &d2c_config).unwrap();
    println!(
        "Depth-to-cloud: {:.1}ms → {} points",
        t2.elapsed().as_secs_f64() * 1000.0,
        cloud.len()
    );

    // Export to PLY
    if cloud.len() > 0 {
        let ply_path = "output_cloud.ply";
        vx_vision::export::write_ply_ascii(ply_path, &cloud).unwrap();
        println!("\nExported point cloud to {}", ply_path);

        if let Some((min, max)) = cloud.bounds() {
            println!(
                "  Bounds: [{:.2}, {:.2}, {:.2}] → [{:.2}, {:.2}, {:.2}]",
                min[0], min[1], min[2], max[0], max[1], max[2]
            );
        }
    }

    println!(
        "\nTotal pipeline: {:.1}ms",
        t0.elapsed().as_secs_f64() * 1000.0
    );
}
