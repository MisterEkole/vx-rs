//! 3D reconstruction data types: point clouds, meshes, depth maps, camera models, and voxel grids.

use crate::error::{Error, Result};
use crate::texture::Texture;
use crate::types::GpuPoint3D;

/// A single 3D point with position, color, and normal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point3D {
    pub position: [f32; 3],
    pub color: [u8; 4],
    pub normal: [f32; 3],
}

impl Default for Point3D {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            color: [255, 255, 255, 255],
            normal: [0.0; 3],
        }
    }
}

/// A mesh vertex with position, normal, and texture coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vertex3D {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

impl Default for Vertex3D {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            normal: [0.0, 1.0, 0.0],
            uv: [0.0; 2],
        }
    }
}

/// A collection of 3D points with optional per-point color and normals.
#[derive(Clone, Debug, Default)]
pub struct PointCloud {
    pub points: Vec<Point3D>,
}

impl PointCloud {
    /// Creates an empty point cloud.
    pub fn new() -> Self {
        Self { points: Vec::new() }
    }

    /// Creates a point cloud with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            points: Vec::with_capacity(capacity),
        }
    }

    /// Creates a point cloud from GPU readback data.
    pub fn from_gpu_points(gpu_points: &[GpuPoint3D]) -> Self {
        let points = gpu_points
            .iter()
            .map(|gp| Point3D {
                position: gp.position,
                color: gp.color,
                normal: gp.normal,
            })
            .collect();
        Self { points }
    }

    /// Returns the number of points.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Returns `true` if the cloud contains no points.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Returns the axis-aligned bounding box as `(min, max)`.
    /// Returns `None` if the cloud is empty.
    pub fn bounds(&self) -> Option<([f32; 3], [f32; 3])> {
        if self.points.is_empty() {
            return None;
        }

        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];

        for p in &self.points {
            for i in 0..3 {
                if p.position[i] < min[i] {
                    min[i] = p.position[i];
                }
                if p.position[i] > max[i] {
                    max[i] = p.position[i];
                }
            }
        }

        Some((min, max))
    }

    /// Returns an iterator over positions only.
    pub fn positions(&self) -> impl Iterator<Item = &[f32; 3]> {
        self.points.iter().map(|p| &p.position)
    }
}

/// A triangle mesh with indexed vertices.
#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub vertices: Vec<Vertex3D>,
    pub indices: Vec<[u32; 3]>,
}

impl Mesh {
    /// Creates an empty mesh.
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
        }
    }

    /// Returns the number of triangular faces.
    pub fn num_faces(&self) -> usize {
        self.indices.len()
    }

    /// Returns the number of vertices.
    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Returns `true` if the mesh has no faces.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Returns the axis-aligned bounding box as `(min, max)`.
    /// Returns `None` if the mesh has no vertices.
    pub fn bounds(&self) -> Option<([f32; 3], [f32; 3])> {
        if self.vertices.is_empty() {
            return None;
        }

        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];

        for v in &self.vertices {
            for i in 0..3 {
                if v.position[i] < min[i] {
                    min[i] = v.position[i];
                }
                if v.position[i] > max[i] {
                    max[i] = v.position[i];
                }
            }
        }

        Some((min, max))
    }

    /// Merges duplicate vertices that share the same quantized position.
    /// Uses a spatial hash with the given tolerance for deduplication.
    pub fn weld_vertices(&mut self, tolerance: f32) {
        use std::collections::HashMap;

        let inv_tol = 1.0 / tolerance;
        let mut vertex_map: HashMap<(i64, i64, i64), u32> = HashMap::new();
        let mut new_vertices: Vec<Vertex3D> = Vec::new();
        let mut remap: Vec<u32> = Vec::with_capacity(self.vertices.len());

        for v in &self.vertices {
            let key = (
                (v.position[0] * inv_tol).round() as i64,
                (v.position[1] * inv_tol).round() as i64,
                (v.position[2] * inv_tol).round() as i64,
            );
            let idx = vertex_map.entry(key).or_insert_with(|| {
                let idx = new_vertices.len() as u32;
                new_vertices.push(*v);
                idx
            });
            remap.push(*idx);
        }

        // Remap indices
        for tri in &mut self.indices {
            tri[0] = remap[tri[0] as usize];
            tri[1] = remap[tri[1] as usize];
            tri[2] = remap[tri[2] as usize];
        }

        // Remove degenerate triangles (where two or more indices are the same)
        self.indices
            .retain(|tri| tri[0] != tri[1] && tri[1] != tri[2] && tri[0] != tri[2]);

        self.vertices = new_vertices;
    }

    /// Computes per-vertex normals by averaging adjacent face normals.
    pub fn compute_normals(&mut self) {
        // Zero out all normals
        for v in &mut self.vertices {
            v.normal = [0.0; 3];
        }

        // Accumulate face normals
        for tri in &self.indices {
            let v0 = self.vertices[tri[0] as usize].position;
            let v1 = self.vertices[tri[1] as usize].position;
            let v2 = self.vertices[tri[2] as usize].position;

            let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
            let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];

            let n = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ];

            for &idx in tri {
                let v = &mut self.vertices[idx as usize];
                v.normal[0] += n[0];
                v.normal[1] += n[1];
                v.normal[2] += n[2];
            }
        }

        // Normalize
        for v in &mut self.vertices {
            let len =
                (v.normal[0] * v.normal[0] + v.normal[1] * v.normal[1] + v.normal[2] * v.normal[2])
                    .sqrt();
            if len > 1e-10 {
                v.normal[0] /= len;
                v.normal[1] /= len;
                v.normal[2] /= len;
            }
        }
    }
}

/// Camera intrinsic parameters (pinhole model).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraIntrinsics {
    /// Focal length in x (pixels).
    pub fx: f32,
    /// Focal length in y (pixels).
    pub fy: f32,
    /// Principal point x (pixels).
    pub cx: f32,
    /// Principal point y (pixels).
    pub cy: f32,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
}

impl CameraIntrinsics {
    /// Creates intrinsics from focal lengths, principal point, and image size.
    pub fn new(fx: f32, fy: f32, cx: f32, cy: f32, width: u32, height: u32) -> Self {
        Self {
            fx,
            fy,
            cx,
            cy,
            width,
            height,
        }
    }

    /// Creates intrinsics with principal point at image center and equal focal lengths.
    pub fn from_focal_length(focal: f32, width: u32, height: u32) -> Self {
        Self {
            fx: focal,
            fy: focal,
            cx: width as f32 / 2.0,
            cy: height as f32 / 2.0,
            width,
            height,
        }
    }

    /// Unprojects a pixel `(u, v)` at depth `d` to a 3D point in camera coordinates.
    pub fn unproject(&self, u: f32, v: f32, depth: f32) -> [f32; 3] {
        [
            (u - self.cx) * depth / self.fx,
            (v - self.cy) * depth / self.fy,
            depth,
        ]
    }

    /// Projects a 3D point in camera coordinates to pixel `(u, v)`.
    pub fn project(&self, point: &[f32; 3]) -> [f32; 2] {
        [
            self.fx * point[0] / point[2] + self.cx,
            self.fy * point[1] / point[2] + self.cy,
        ]
    }
}

/// Camera extrinsic parameters: rotation (row-major 3x3) and translation.
///
/// Transforms a point from world coordinates to camera coordinates:
/// `p_cam = R * p_world + t`
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraExtrinsics {
    /// Row-major 3x3 rotation matrix.
    pub rotation: [f32; 9],
    /// Translation vector.
    pub translation: [f32; 3],
}

impl CameraExtrinsics {
    /// Creates extrinsics from rotation matrix and translation.
    pub fn new(rotation: [f32; 9], translation: [f32; 3]) -> Self {
        Self {
            rotation,
            translation,
        }
    }

    /// Identity transform (no rotation, no translation).
    pub fn identity() -> Self {
        Self {
            rotation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            translation: [0.0; 3],
        }
    }

    /// Transforms a world-space point to camera coordinates.
    pub fn transform_point(&self, p: &[f32; 3]) -> [f32; 3] {
        let r = &self.rotation;
        let t = &self.translation;
        [
            r[0] * p[0] + r[1] * p[1] + r[2] * p[2] + t[0],
            r[3] * p[0] + r[4] * p[1] + r[5] * p[2] + t[1],
            r[6] * p[0] + r[7] * p[1] + r[8] * p[2] + t[2],
        ]
    }

    /// Returns the inverse transform (camera-to-world).
    pub fn inverse(&self) -> Self {
        let r = &self.rotation;
        let t = &self.translation;
        // R_inv = R^T
        let r_inv = [r[0], r[3], r[6], r[1], r[4], r[7], r[2], r[5], r[8]];
        // t_inv = -R^T * t
        let t_inv = [
            -(r_inv[0] * t[0] + r_inv[1] * t[1] + r_inv[2] * t[2]),
            -(r_inv[3] * t[0] + r_inv[4] * t[1] + r_inv[5] * t[2]),
            -(r_inv[6] * t[0] + r_inv[7] * t[1] + r_inv[8] * t[2]),
        ];
        Self {
            rotation: r_inv,
            translation: t_inv,
        }
    }

    /// Packs rotation and translation into the GPU-compatible format:
    /// three `[f32; 4]` rows where `.xyz` = rotation row, `.w` = translation component.
    pub fn to_gpu_rows(&self) -> [[f32; 4]; 3] {
        let r = &self.rotation;
        let t = &self.translation;
        [
            [r[0], r[1], r[2], t[0]],
            [r[3], r[4], r[5], t[1]],
            [r[6], r[7], r[8], t[2]],
        ]
    }
}

/// A depth map: an R32Float texture with associated camera intrinsics and depth range.
pub struct DepthMap {
    texture: Texture,
    intrinsics: CameraIntrinsics,
    min_depth: f32,
    max_depth: f32,
}

impl DepthMap {
    /// Creates a depth map from an R32Float texture.
    pub fn new(
        texture: Texture,
        intrinsics: CameraIntrinsics,
        min_depth: f32,
        max_depth: f32,
    ) -> Result<Self> {
        if texture.format() != crate::texture::TextureFormat::R32Float {
            return Err(Error::InvalidConfig(
                "DepthMap requires an R32Float texture".into(),
            ));
        }
        Ok(Self {
            texture,
            intrinsics,
            min_depth,
            max_depth,
        })
    }

    /// Returns a reference to the underlying depth texture.
    pub fn texture(&self) -> &Texture {
        &self.texture
    }

    /// Returns the camera intrinsics.
    pub fn intrinsics(&self) -> &CameraIntrinsics {
        &self.intrinsics
    }

    /// Returns the minimum valid depth.
    pub fn min_depth(&self) -> f32 {
        self.min_depth
    }

    /// Returns the maximum valid depth.
    pub fn max_depth(&self) -> f32 {
        self.max_depth
    }

    /// Width in pixels.
    pub fn width(&self) -> u32 {
        self.texture.width()
    }

    /// Height in pixels.
    pub fn height(&self) -> u32 {
        self.texture.height()
    }

    /// Reads back depth values and unprojects valid pixels to a point cloud (CPU-side).
    /// For GPU-accelerated unprojection, use the `DepthToCloud` kernel.
    pub fn to_point_cloud(&self) -> PointCloud {
        let data = self.texture.read_r32float();
        let w = self.width() as usize;
        let h = self.height() as usize;
        let mut points = Vec::new();

        for y in 0..h {
            for x in 0..w {
                let d = data[y * w + x];
                if d >= self.min_depth && d <= self.max_depth {
                    let pos = self.intrinsics.unproject(x as f32, y as f32, d);
                    points.push(Point3D {
                        position: pos,
                        color: [255, 255, 255, 255],
                        normal: [0.0; 3],
                    });
                }
            }
        }

        PointCloud { points }
    }
}

/// A 3D voxel grid for TSDF (Truncated Signed Distance Function) storage.
///
/// Each voxel stores an SDF value and a weight. The grid is backed by
/// [`UnifiedBuffer`](vx_gpu::UnifiedBuffer) for zero-copy GPU access.
pub struct VoxelGrid {
    /// TSDF values per voxel (initialized to 1.0 = far from surface).
    pub tsdf: vx_gpu::UnifiedBuffer<f32>,
    /// Integration weights per voxel (initialized to 0.0 = not observed).
    pub weights: vx_gpu::UnifiedBuffer<f32>,
    /// Number of voxels in each dimension `[x, y, z]`.
    pub resolution: [u32; 3],
    /// Size of each voxel in meters.
    pub voxel_size: f32,
    /// World-space origin of the grid (corner with smallest coordinates).
    pub origin: [f32; 3],
}

impl VoxelGrid {
    /// Allocates a new voxel grid on the given Metal device.
    ///
    /// All TSDF values are initialized to `1.0` (far from any surface)
    /// and all weights to `0.0` (unobserved).
    pub fn new(
        device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
        resolution: [u32; 3],
        voxel_size: f32,
        origin: [f32; 3],
    ) -> std::result::Result<Self, String> {
        let total = resolution[0] as usize * resolution[1] as usize * resolution[2] as usize;

        let mut tsdf = vx_gpu::UnifiedBuffer::<f32>::new(device, total)?;
        let weights = vx_gpu::UnifiedBuffer::<f32>::new(device, total)?;

        // Initialize TSDF to 1.0 (far from surface)
        let init_vals: Vec<f32> = vec![1.0; total];
        tsdf.write(&init_vals);

        Ok(Self {
            tsdf,
            weights,
            resolution,
            voxel_size,
            origin,
        })
    }

    /// Allocates a new voxel grid using a [`Context`](crate::Context).
    pub fn from_context(
        ctx: &crate::context::Context,
        resolution: [u32; 3],
        voxel_size: f32,
        origin: [f32; 3],
    ) -> crate::error::Result<Self> {
        Self::new(ctx.device(), resolution, voxel_size, origin).map_err(crate::error::Error::from)
    }

    /// Total number of voxels.
    pub fn num_voxels(&self) -> usize {
        self.resolution[0] as usize * self.resolution[1] as usize * self.resolution[2] as usize
    }

    /// World-space extent of the grid in each dimension.
    pub fn extent(&self) -> [f32; 3] {
        [
            self.resolution[0] as f32 * self.voxel_size,
            self.resolution[1] as f32 * self.voxel_size,
            self.resolution[2] as f32 * self.voxel_size,
        ]
    }

    /// Converts a voxel index `(ix, iy, iz)` to world-space coordinates (voxel center).
    pub fn voxel_to_world(&self, ix: u32, iy: u32, iz: u32) -> [f32; 3] {
        [
            self.origin[0] + (ix as f32 + 0.5) * self.voxel_size,
            self.origin[1] + (iy as f32 + 0.5) * self.voxel_size,
            self.origin[2] + (iz as f32 + 0.5) * self.voxel_size,
        ]
    }

    /// Returns the flat buffer index for voxel `(ix, iy, iz)`.
    pub fn voxel_index(&self, ix: u32, iy: u32, iz: u32) -> usize {
        (iz as usize * self.resolution[1] as usize + iy as usize) * self.resolution[0] as usize
            + ix as usize
    }

    /// Resets the grid to its initial state (TSDF = 1.0, weights = 0.0).
    pub fn reset(&mut self) {
        let total = self.num_voxels();
        let init_tsdf: Vec<f32> = vec![1.0; total];
        let init_weights: Vec<f32> = vec![0.0; total];
        self.tsdf.write(&init_tsdf);
        self.weights.write(&init_weights);
    }
}

/// Colormap for depth visualization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Colormap {
    /// Turbo colormap (perceptually smooth rainbow).
    Turbo,
    /// Jet colormap (blue-cyan-green-yellow-red).
    Jet,
    /// Inferno colormap (dark purple to yellow).
    Inferno,
}

impl Default for Colormap {
    fn default() -> Self {
        Self::Turbo
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_cloud_bounds() {
        let cloud = PointCloud {
            points: vec![
                Point3D {
                    position: [1.0, 2.0, 3.0],
                    ..Default::default()
                },
                Point3D {
                    position: [-1.0, 5.0, 0.0],
                    ..Default::default()
                },
                Point3D {
                    position: [3.0, -1.0, 2.0],
                    ..Default::default()
                },
            ],
        };

        let (min, max) = cloud.bounds().unwrap();
        assert_eq!(min, [-1.0, -1.0, 0.0]);
        assert_eq!(max, [3.0, 5.0, 3.0]);
    }

    #[test]
    fn empty_cloud_bounds() {
        let cloud = PointCloud::new();
        assert!(cloud.bounds().is_none());
    }

    #[test]
    fn mesh_compute_normals() {
        // A single triangle in the XY plane: normal should point in +Z
        let mut mesh = Mesh {
            vertices: vec![
                Vertex3D {
                    position: [0.0, 0.0, 0.0],
                    ..Default::default()
                },
                Vertex3D {
                    position: [1.0, 0.0, 0.0],
                    ..Default::default()
                },
                Vertex3D {
                    position: [0.0, 1.0, 0.0],
                    ..Default::default()
                },
            ],
            indices: vec![[0, 1, 2]],
        };

        mesh.compute_normals();

        for v in &mesh.vertices {
            assert!((v.normal[0]).abs() < 1e-5);
            assert!((v.normal[1]).abs() < 1e-5);
            assert!((v.normal[2] - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn mesh_weld_vertices() {
        let mut mesh = Mesh {
            vertices: vec![
                Vertex3D {
                    position: [0.0, 0.0, 0.0],
                    ..Default::default()
                },
                Vertex3D {
                    position: [1.0, 0.0, 0.0],
                    ..Default::default()
                },
                Vertex3D {
                    position: [0.0, 1.0, 0.0],
                    ..Default::default()
                },
                // Duplicate of vertex 0
                Vertex3D {
                    position: [0.001, 0.001, 0.0],
                    ..Default::default()
                },
            ],
            indices: vec![[0, 1, 2], [3, 1, 2]],
        };

        mesh.weld_vertices(0.01);
        assert!(mesh.num_vertices() <= 3);
    }

    #[test]
    fn camera_intrinsics_project_unproject_roundtrip() {
        let cam = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
        let point = [0.5, -0.3, 2.0];
        let pixel = cam.project(&point);
        let back = cam.unproject(pixel[0], pixel[1], point[2]);

        for i in 0..3 {
            assert!((back[i] - point[i]).abs() < 1e-4, "axis {i} mismatch");
        }
    }

    #[test]
    fn camera_extrinsics_identity() {
        let ext = CameraExtrinsics::identity();
        let p = [1.0, 2.0, 3.0];
        let result = ext.transform_point(&p);
        assert_eq!(result, p);
    }

    #[test]
    fn camera_extrinsics_inverse_roundtrip() {
        let ext = CameraExtrinsics::new(
            [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0], // 90° rotation around Z
            [1.0, 2.0, 3.0],
        );
        let inv = ext.inverse();
        let p = [5.0, 6.0, 7.0];
        let forward = ext.transform_point(&p);
        let back = inv.transform_point(&forward);

        for i in 0..3 {
            assert!(
                (back[i] - p[i]).abs() < 1e-4,
                "axis {i}: {} vs {}",
                back[i],
                p[i]
            );
        }
    }

    #[test]
    fn gpu_point_conversion() {
        let gpu_points = vec![GpuPoint3D {
            position: [1.0, 2.0, 3.0],
            _pad0: 0.0,
            color: [128, 64, 32, 255],
            normal: [0.0, 1.0, 0.0],
            _pad1: 0.0,
        }];

        let cloud = PointCloud::from_gpu_points(&gpu_points);
        assert_eq!(cloud.len(), 1);
        assert_eq!(cloud.points[0].position, [1.0, 2.0, 3.0]);
        assert_eq!(cloud.points[0].color, [128, 64, 32, 255]);
    }

    #[test]
    fn colormap_default() {
        assert_eq!(Colormap::default(), Colormap::Turbo);
    }
}
