// examples/tsdf_fusion_demo.rs
//
// TSDF fusion of synthetic depth frames -> marching cubes -> render + export mesh and cloud.
//
// Run with:
//   cargo run --release --example tsdf_fusion_demo --features full

use std::time::Instant;

use vx_vision::kernels::depth_colorize::{DepthColorize, DepthColorizeConfig};
use vx_vision::kernels::marching_cubes::{MarchingCubes, MarchingCubesConfig};
use vx_vision::kernels::tsdf::{TSDFConfig, TSDFVolume};
use vx_vision::render_context::{Camera, RenderTarget};
use vx_vision::renderers::mesh_renderer::MeshRenderer;
use vx_vision::renderers::point_cloud_renderer::PointCloudRenderer;
use vx_vision::types_3d::{CameraExtrinsics, CameraIntrinsics, DepthMap};
use vx_vision::Context;

fn save_rgba_png(path: &str, pixels: &[u8], width: u32, height: u32) {
    let img = image::RgbaImage::from_raw(width, height, pixels.to_vec())
        .expect("pixel buffer size mismatch");
    img.save(path).expect("failed to save PNG");
}

fn main() {
    let ctx = Context::new().expect("No Metal GPU available");
    println!("VX TSDF Fusion + Rendering Demo");
    println!("================================\n");

    // Configure TSDF volume
    let mut config = TSDFConfig::default();
    config.resolution = [128, 128, 128];
    config.voxel_size = 0.005;
    config.origin = [-0.32, -0.32, 0.0];
    config.truncation_distance = 0.02;
    config.max_weight = 50.0;

    let t0 = Instant::now();
    let tsdf = TSDFVolume::new(&ctx, config).unwrap();
    println!(
        "TSDF volume: 128^3 x {:.3}m voxels ({:.1}MB)",
        0.005,
        128.0 * 128.0 * 128.0 * 8.0 / 1e6
    );

    // Camera intrinsics (shared for depth generation and rendering)
    let depth_w = 128u32;
    let depth_h = 128u32;
    let intrinsics = CameraIntrinsics::new(300.0, 300.0, 64.0, 64.0, depth_w, depth_h);

    // Sphere parameters
    let sphere_center = [0.0f32, 0.0, 0.32];
    let sphere_radius = 0.15f32;

    // Generate + integrate depth frames
    let colorize = DepthColorize::new(&ctx).unwrap();
    let render_w = 512u32;
    let render_h = 512u32;

    println!("\nIntegrating depth frames...");

    let n_views = 6;
    for view in 0..n_views {
        // Rotate camera around the sphere
        let angle = view as f32 * std::f32::consts::PI * 2.0 / n_views as f32 * 0.3;
        let cam_dist = 0.5f32;

        let cam_x = angle.sin() * 0.1;
        let cam_y = 0.0f32;

        // Camera looking along +Z with slight horizontal offset
        let pose = CameraExtrinsics::new(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [-cam_x, -cam_y, 0.0],
        );

        // Raytrace synthetic depth
        let depth_data: Vec<f32> = (0..(depth_w * depth_h))
            .map(|i| {
                let px = (i % depth_w) as f32;
                let py = (i / depth_w) as f32;

                let dx = (px - intrinsics.cx) / intrinsics.fx;
                let dy = (py - intrinsics.cy) / intrinsics.fy;
                let dz = 1.0f32;
                let len = (dx * dx + dy * dy + dz * dz).sqrt();
                let ray = [dx / len, dy / len, dz / len];

                // Camera origin in world = inverse of translation
                let origin = [cam_x, cam_y, 0.0];

                let oc = [
                    origin[0] - sphere_center[0],
                    origin[1] - sphere_center[1],
                    origin[2] - sphere_center[2],
                ];
                let a = ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2];
                let b = 2.0 * (oc[0] * ray[0] + oc[1] * ray[1] + oc[2] * ray[2]);
                let c =
                    oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2] - sphere_radius * sphere_radius;
                let disc = b * b - 4.0 * a * c;

                if disc > 0.0 {
                    let t = (-b - disc.sqrt()) / (2.0 * a);
                    if t > 0.01 {
                        t
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            })
            .collect();

        let valid = depth_data.iter().filter(|&&d| d > 0.0).count();
        let depth_tex = ctx.texture_r32float(&depth_data, depth_w, depth_h).unwrap();
        let depth_map = DepthMap::new(depth_tex, intrinsics, 0.01, 2.0).unwrap();

        tsdf.integrate(&ctx, &depth_map, &pose).unwrap();

        // Save first depth frame as colorized PNG
        if view == 0 {
            let color_out = ctx.texture_output_rgba8(depth_w, depth_h).unwrap();
            let color_config = DepthColorizeConfig::new(0.1, 0.5);
            let depth_vis = ctx.texture_r32float(&depth_data, depth_w, depth_h).unwrap();
            colorize
                .apply(&ctx, &depth_vis, &color_out, &color_config)
                .unwrap();
            let pixels = color_out.read_rgba8();
            save_rgba_png("depth_colorized.png", &pixels, depth_w, depth_h);
            println!("  Saved depth_colorized.png ({}x{})", depth_w, depth_h);
        }

        println!("  View {}: {} valid depth pixels", view, valid);
    }

    // Extract surface
    let t_ext = Instant::now();
    let cloud = tsdf.extract_cloud();
    println!(
        "\nSurface extraction: {:.1}ms -> {} points",
        t_ext.elapsed().as_secs_f64() * 1000.0,
        cloud.len()
    );

    // Marching Cubes mesh
    let t_mc = Instant::now();
    let mc_config = MarchingCubesConfig::default();
    let mut mesh = MarchingCubes::extract(tsdf.volume(), &mc_config);
    mesh.compute_normals();
    mesh.weld_vertices(0.001);
    println!(
        "Marching Cubes: {:.1}ms -> {} vertices, {} triangles",
        t_mc.elapsed().as_secs_f64() * 1000.0,
        mesh.num_vertices(),
        mesh.num_faces()
    );

    // Render point cloud to PNG
    if cloud.len() > 0 {
        let pc_renderer = PointCloudRenderer::new(&ctx).unwrap();
        let target = RenderTarget::new(&ctx, render_w, render_h).unwrap();

        // Position camera to see the sphere (it's at z=0.32)
        let camera = Camera {
            position: [0.3, 0.2, -0.1],
            look_at: [0.0, 0.0, 0.32],
            up: [0.0, 1.0, 0.0],
            fov_y: 50.0_f32.to_radians(),
            near: 0.01,
            far: 10.0,
        };

        pc_renderer
            .render(&ctx, &cloud, &camera, &target, 4.0)
            .unwrap();
        let pixels = target.read_rgba8();
        save_rgba_png("point_cloud_render.png", &pixels, render_w, render_h);
        println!("\nSaved point_cloud_render.png ({}x{})", render_w, render_h);
    }

    // Render mesh to PNG
    if mesh.num_faces() > 0 {
        let mesh_renderer = MeshRenderer::new(&ctx).unwrap();
        let target = RenderTarget::new(&ctx, render_w, render_h).unwrap();

        let camera = Camera {
            position: [0.3, 0.2, -0.1],
            look_at: [0.0, 0.0, 0.32],
            up: [0.0, 1.0, 0.0],
            fov_y: 50.0_f32.to_radians(),
            near: 0.01,
            far: 10.0,
        };

        mesh_renderer.render(&ctx, &mesh, &camera, &target).unwrap();
        let pixels = target.read_rgba8();
        save_rgba_png("mesh_render.png", &pixels, render_w, render_h);
        println!("Saved mesh_render.png ({}x{})", render_w, render_h);
    }

    // Export files
    if mesh.num_faces() > 0 {
        vx_vision::export::write_obj("tsdf_sphere.obj", &mesh).unwrap();
        println!(
            "\nExported tsdf_sphere.obj ({} triangles)",
            mesh.num_faces()
        );
    }
    if cloud.len() > 0 {
        vx_vision::export::write_ply_ascii("tsdf_sphere.ply", &cloud).unwrap();
        println!("Exported tsdf_sphere.ply ({} points)", cloud.len());
    }

    if let Some((min, max)) = mesh.bounds() {
        println!(
            "\nMesh bounds: [{:.3}, {:.3}, {:.3}] -> [{:.3}, {:.3}, {:.3}]",
            min[0], min[1], min[2], max[0], max[1], max[2]
        );
    }

    println!(
        "\nTotal pipeline: {:.0}ms",
        t0.elapsed().as_secs_f64() * 1000.0
    );
    println!("\nOutput files:");
    println!("  depth_colorized.png   - colorized depth map");
    println!("  point_cloud_render.png - rendered point cloud");
    println!("  mesh_render.png        - rendered mesh with Phong shading");
    println!("  tsdf_sphere.obj        - triangle mesh (OBJ)");
    println!("  tsdf_sphere.ply        - point cloud (PLY)");
}
