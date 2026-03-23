# Visualization

Requires the `visualization` feature flag (and `reconstruction` for 3D types): `cargo build --features "reconstruction,visualization"`

## Point Cloud Renderer

Renders point clouds as colored circle splats using Metal render pipelines. Outputs to an offscreen `RenderTarget`.

```rust
use vx_vision::render_context::{Camera, RenderTarget};
use vx_vision::renderers::point_cloud_renderer::PointCloudRenderer;

let renderer = PointCloudRenderer::new(&ctx)?;
let target = RenderTarget::new(&ctx, 1920, 1080)?;
let camera = Camera {
    position: [0.0, 0.0, 3.0],
    look_at: [0.0, 0.0, 0.0],
    up: [0.0, 1.0, 0.0],
    fov_y: 60.0_f32.to_radians(),
    near: 0.01,
    far: 100.0,
};

renderer.render(&ctx, &cloud, &camera, &target, 5.0)?;

// Read back as RGBA pixels
let pixels = target.read_rgba8();
```

**Parameters:** `point_size` controls the rendered diameter of each point in pixels.

## Mesh Renderer

Renders triangle meshes with Phong shading (ambient + diffuse). Default light direction is (0.5, 0.7, 1.0).

```rust
use vx_vision::render_context::{Camera, RenderTarget};
use vx_vision::renderers::mesh_renderer::MeshRenderer;

let renderer = MeshRenderer::new(&ctx)?;
let target = RenderTarget::new(&ctx, 1920, 1080)?;
let camera = Camera::default();

renderer.render(&ctx, &mesh, &camera, &target)?;
let pixels = target.read_rgba8();
```

## RenderTarget

Offscreen render target with RGBA8 color and Depth32Float attachments.

```rust
use vx_vision::render_context::RenderTarget;

let target = RenderTarget::new(&ctx, width, height)?;

// After rendering:
let rgba_pixels = target.read_rgba8();   // Vec<u8>, 4 bytes per pixel
let color_tex = target.color_texture();  // &Texture for further processing
```

## Camera

Camera parameters for 3D rendering. Computes the MVP (model-view-projection) matrix.

```rust
use vx_vision::render_context::Camera;

let camera = Camera {
    position: [2.0, 1.5, 3.0],
    look_at: [0.0, 0.0, 0.0],
    up: [0.0, 1.0, 0.0],
    fov_y: 45.0_f32.to_radians(),
    near: 0.1,
    far: 50.0,
};

let mvp = camera.mvp_matrix(width as f32 / height as f32);
```

## Depth Colorize

See [3D Reconstruction → Depth Colorize](reconstruction.md#depth-colorize) — available with only the `reconstruction` feature.
