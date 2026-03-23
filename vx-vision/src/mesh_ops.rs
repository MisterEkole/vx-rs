//! CPU-side mesh processing utilities: decimation and cleanup.

#[cfg(feature = "reconstruction")]
use crate::types_3d::Mesh;

/// Decimates a mesh by iteratively collapsing the shortest edges.
///
/// This is a simplified edge-collapse decimation. For production use,
/// quadric error metrics (QEM) would produce better results.
#[cfg(feature = "reconstruction")]
pub fn decimate(mesh: &Mesh, target_faces: usize) -> Mesh {
    if mesh.num_faces() <= target_faces {
        return mesh.clone();
    }

    let mut result = mesh.clone();

    // Simple edge-length-based decimation:
    // Repeatedly find and collapse the shortest edge until target is reached
    while result.num_faces() > target_faces && result.num_faces() > 4 {
        // Find shortest edge
        let mut best_edge = (0u32, 0u32);
        let mut best_len2 = f32::MAX;

        for tri in &result.indices {
            let edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])];
            for (a, b) in edges {
                let va = result.vertices[a as usize].position;
                let vb = result.vertices[b as usize].position;
                let dx = va[0] - vb[0];
                let dy = va[1] - vb[1];
                let dz = va[2] - vb[2];
                let len2 = dx * dx + dy * dy + dz * dz;
                if len2 < best_len2 {
                    best_len2 = len2;
                    best_edge = (a, b);
                }
            }
        }

        let (a, b) = best_edge;
        if a == b {
            break;
        }

        // Collapse: move vertex `a` to midpoint, replace all references to `b` with `a`
        let va = result.vertices[a as usize].position;
        let vb = result.vertices[b as usize].position;
        result.vertices[a as usize].position = [
            (va[0] + vb[0]) * 0.5,
            (va[1] + vb[1]) * 0.5,
            (va[2] + vb[2]) * 0.5,
        ];

        // Replace b → a in all triangles
        for tri in &mut result.indices {
            for idx in tri.iter_mut() {
                if *idx == b {
                    *idx = a;
                }
            }
        }

        // Remove degenerate triangles
        result
            .indices
            .retain(|tri| tri[0] != tri[1] && tri[1] != tri[2] && tri[0] != tri[2]);
    }

    result
}
