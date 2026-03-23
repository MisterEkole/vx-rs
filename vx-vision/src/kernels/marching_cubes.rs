//! Marching Cubes mesh extraction from TSDF volumes.

#[cfg(feature = "reconstruction")]
use crate::types_3d::{Mesh, Vertex3D, VoxelGrid};

/// Configuration for marching cubes extraction.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct MarchingCubesConfig {
    /// Iso-level for surface extraction (typically 0.0 for TSDF).
    pub iso_level: f32,
}

impl Default for MarchingCubesConfig {
    fn default() -> Self {
        Self { iso_level: 0.0 }
    }
}

// Corner offsets for the 8 corners of a cube
const CORNER_OFFSETS: [[usize; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
];

// Edge-to-corner vertex pairs (12 edges)
const EDGE_VERTICES: [[usize; 2]; 12] = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
];

// Edge table: for each of 256 cube configs, which edges are intersected (bitmask)
#[rustfmt::skip]
const EDGE_TABLE: [u16; 256] = [
    0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x055, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x066, 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x033, 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000,
];

// Triangle table (first 64 entries shown; the full table has 256 entries)
// Each entry lists up to 15 edge indices (5 triangles), terminated by -1
// This is the standard Lorensen & Cline table
#[rustfmt::skip]
const TRI_TABLE: [[i8; 16]; 256] = include!("mc_tri_table.inc");

/// Marching Cubes mesh extractor (CPU-side, reads from UMA shared buffers).
pub struct MarchingCubes;

#[cfg(feature = "reconstruction")]
impl MarchingCubes {
    /// Extracts a triangle mesh from a voxel grid at the given iso-level.
    pub fn extract(grid: &VoxelGrid, config: &MarchingCubesConfig) -> Mesh {
        let res = grid.resolution;
        let tsdf = grid.tsdf.as_slice();
        let vs = grid.voxel_size;
        let iso = config.iso_level;

        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for iz in 0..res[2] as usize - 1 {
            for iy in 0..res[1] as usize - 1 {
                for ix in 0..res[0] as usize - 1 {
                    // Read 8 corner values
                    let mut vals = [0.0f32; 8];
                    let mut positions = [[0.0f32; 3]; 8];

                    for (c, off) in CORNER_OFFSETS.iter().enumerate() {
                        let cx = ix + off[0];
                        let cy = iy + off[1];
                        let cz = iz + off[2];
                        let idx = (cz * res[1] as usize + cy) * res[0] as usize + cx;
                        vals[c] = tsdf[idx];
                        positions[c] = [
                            grid.origin[0] + (cx as f32 + 0.5) * vs,
                            grid.origin[1] + (cy as f32 + 0.5) * vs,
                            grid.origin[2] + (cz as f32 + 0.5) * vs,
                        ];
                    }

                    // Compute cube index
                    let mut cube_index = 0u8;
                    for c in 0..8 {
                        if vals[c] < iso {
                            cube_index |= 1 << c;
                        }
                    }

                    let edges = EDGE_TABLE[cube_index as usize];
                    if edges == 0 {
                        continue;
                    }

                    // Interpolate vertices on edges
                    let mut edge_verts = [[0.0f32; 3]; 12];
                    for e in 0..12 {
                        if edges & (1 << e) != 0 {
                            let [c0, c1] = EDGE_VERTICES[e];
                            let t = if (vals[c1] - vals[c0]).abs() > 1e-10 {
                                (iso - vals[c0]) / (vals[c1] - vals[c0])
                            } else {
                                0.5
                            };
                            let t = t.clamp(0.0, 1.0);
                            for i in 0..3 {
                                edge_verts[e][i] =
                                    positions[c0][i] + t * (positions[c1][i] - positions[c0][i]);
                            }
                        }
                    }

                    // Generate triangles
                    let row = TRI_TABLE[cube_index as usize];
                    let mut i = 0;
                    while i < 16 && row[i] >= 0 {
                        let e0 = row[i] as usize;
                        let e1 = row[i + 1] as usize;
                        let e2 = row[i + 2] as usize;

                        let base = vertices.len() as u32;

                        // Compute face normal
                        let v0 = edge_verts[e0];
                        let v1 = edge_verts[e1];
                        let v2 = edge_verts[e2];
                        let a = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
                        let b = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
                        let mut n = [
                            a[1] * b[2] - a[2] * b[1],
                            a[2] * b[0] - a[0] * b[2],
                            a[0] * b[1] - a[1] * b[0],
                        ];
                        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                        if len > 1e-10 {
                            n[0] /= len;
                            n[1] /= len;
                            n[2] /= len;
                        }

                        vertices.push(Vertex3D {
                            position: v0,
                            normal: n,
                            uv: [0.0; 2],
                        });
                        vertices.push(Vertex3D {
                            position: v1,
                            normal: n,
                            uv: [0.0; 2],
                        });
                        vertices.push(Vertex3D {
                            position: v2,
                            normal: n,
                            uv: [0.0; 2],
                        });

                        indices.push([base, base + 1, base + 2]);

                        i += 3;
                    }
                }
            }
        }

        Mesh { vertices, indices }
    }
}
