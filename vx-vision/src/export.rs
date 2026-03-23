//! Export point clouds and meshes to standard 3D file formats.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::error::{Error, Result};
use crate::types_3d::{Mesh, PointCloud};

/// Writes a point cloud to a PLY file in ASCII format.
pub fn write_ply_ascii<P: AsRef<Path>>(path: P, cloud: &PointCloud) -> Result<()> {
    let file = File::create(path.as_ref())
        .map_err(|e| Error::InvalidConfig(format!("cannot create file: {e}")))?;
    let mut w = BufWriter::new(file);

    let has_normals = cloud
        .points
        .iter()
        .any(|p| p.normal[0] != 0.0 || p.normal[1] != 0.0 || p.normal[2] != 0.0);

    writeln!(w, "ply").map_err(io_err)?;
    writeln!(w, "format ascii 1.0").map_err(io_err)?;
    writeln!(w, "element vertex {}", cloud.len()).map_err(io_err)?;
    writeln!(w, "property float x").map_err(io_err)?;
    writeln!(w, "property float y").map_err(io_err)?;
    writeln!(w, "property float z").map_err(io_err)?;
    writeln!(w, "property uchar red").map_err(io_err)?;
    writeln!(w, "property uchar green").map_err(io_err)?;
    writeln!(w, "property uchar blue").map_err(io_err)?;
    writeln!(w, "property uchar alpha").map_err(io_err)?;
    if has_normals {
        writeln!(w, "property float nx").map_err(io_err)?;
        writeln!(w, "property float ny").map_err(io_err)?;
        writeln!(w, "property float nz").map_err(io_err)?;
    }
    writeln!(w, "end_header").map_err(io_err)?;

    for p in &cloud.points {
        if has_normals {
            writeln!(
                w,
                "{} {} {} {} {} {} {} {} {} {}",
                p.position[0],
                p.position[1],
                p.position[2],
                p.color[0],
                p.color[1],
                p.color[2],
                p.color[3],
                p.normal[0],
                p.normal[1],
                p.normal[2],
            )
            .map_err(io_err)?;
        } else {
            writeln!(
                w,
                "{} {} {} {} {} {} {}",
                p.position[0],
                p.position[1],
                p.position[2],
                p.color[0],
                p.color[1],
                p.color[2],
                p.color[3],
            )
            .map_err(io_err)?;
        }
    }

    w.flush().map_err(io_err)?;
    Ok(())
}

/// Writes a point cloud to a PLY file in binary little-endian format.
pub fn write_ply_binary<P: AsRef<Path>>(path: P, cloud: &PointCloud) -> Result<()> {
    let file = File::create(path.as_ref())
        .map_err(|e| Error::InvalidConfig(format!("cannot create file: {e}")))?;
    let mut w = BufWriter::new(file);

    let has_normals = cloud
        .points
        .iter()
        .any(|p| p.normal[0] != 0.0 || p.normal[1] != 0.0 || p.normal[2] != 0.0);

    // Write ASCII header
    writeln!(w, "ply").map_err(io_err)?;
    writeln!(w, "format binary_little_endian 1.0").map_err(io_err)?;
    writeln!(w, "element vertex {}", cloud.len()).map_err(io_err)?;
    writeln!(w, "property float x").map_err(io_err)?;
    writeln!(w, "property float y").map_err(io_err)?;
    writeln!(w, "property float z").map_err(io_err)?;
    writeln!(w, "property uchar red").map_err(io_err)?;
    writeln!(w, "property uchar green").map_err(io_err)?;
    writeln!(w, "property uchar blue").map_err(io_err)?;
    writeln!(w, "property uchar alpha").map_err(io_err)?;
    if has_normals {
        writeln!(w, "property float nx").map_err(io_err)?;
        writeln!(w, "property float ny").map_err(io_err)?;
        writeln!(w, "property float nz").map_err(io_err)?;
    }
    writeln!(w, "end_header").map_err(io_err)?;

    // Write binary data
    for p in &cloud.points {
        w.write_all(&p.position[0].to_le_bytes()).map_err(io_err)?;
        w.write_all(&p.position[1].to_le_bytes()).map_err(io_err)?;
        w.write_all(&p.position[2].to_le_bytes()).map_err(io_err)?;
        w.write_all(&p.color).map_err(io_err)?;
        if has_normals {
            w.write_all(&p.normal[0].to_le_bytes()).map_err(io_err)?;
            w.write_all(&p.normal[1].to_le_bytes()).map_err(io_err)?;
            w.write_all(&p.normal[2].to_le_bytes()).map_err(io_err)?;
        }
    }

    w.flush().map_err(io_err)?;
    Ok(())
}

/// Writes a triangle mesh to an OBJ file.
pub fn write_obj<P: AsRef<Path>>(path: P, mesh: &Mesh) -> Result<()> {
    let file = File::create(path.as_ref())
        .map_err(|e| Error::InvalidConfig(format!("cannot create file: {e}")))?;
    let mut w = BufWriter::new(file);

    writeln!(w, "# VX-RS mesh export").map_err(io_err)?;
    writeln!(w, "# Vertices: {}", mesh.num_vertices()).map_err(io_err)?;
    writeln!(w, "# Faces: {}", mesh.num_faces()).map_err(io_err)?;

    let has_normals = mesh
        .vertices
        .iter()
        .any(|v| v.normal[0] != 0.0 || v.normal[1] != 0.0 || v.normal[2] != 0.0);

    let has_uvs = mesh
        .vertices
        .iter()
        .any(|v| v.uv[0] != 0.0 || v.uv[1] != 0.0);

    // Vertices
    for v in &mesh.vertices {
        writeln!(w, "v {} {} {}", v.position[0], v.position[1], v.position[2]).map_err(io_err)?;
    }

    // Normals
    if has_normals {
        for v in &mesh.vertices {
            writeln!(w, "vn {} {} {}", v.normal[0], v.normal[1], v.normal[2]).map_err(io_err)?;
        }
    }

    // Texture coordinates
    if has_uvs {
        for v in &mesh.vertices {
            writeln!(w, "vt {} {}", v.uv[0], v.uv[1]).map_err(io_err)?;
        }
    }

    // Faces (OBJ uses 1-based indices)
    for tri in &mesh.indices {
        let (i0, i1, i2) = (tri[0] + 1, tri[1] + 1, tri[2] + 1);
        match (has_normals, has_uvs) {
            (true, true) => {
                writeln!(w, "f {i0}/{i0}/{i0} {i1}/{i1}/{i1} {i2}/{i2}/{i2}").map_err(io_err)?
            }
            (true, false) => writeln!(w, "f {i0}//{i0} {i1}//{i1} {i2}//{i2}").map_err(io_err)?,
            (false, true) => writeln!(w, "f {i0}/{i0} {i1}/{i1} {i2}/{i2}").map_err(io_err)?,
            (false, false) => writeln!(w, "f {i0} {i1} {i2}").map_err(io_err)?,
        }
    }

    w.flush().map_err(io_err)?;
    Ok(())
}

/// Writes a point cloud as a PLY file with mesh faces (for colored meshes).
pub fn write_mesh_ply<P: AsRef<Path>>(path: P, mesh: &Mesh) -> Result<()> {
    let file = File::create(path.as_ref())
        .map_err(|e| Error::InvalidConfig(format!("cannot create file: {e}")))?;
    let mut w = BufWriter::new(file);

    writeln!(w, "ply").map_err(io_err)?;
    writeln!(w, "format ascii 1.0").map_err(io_err)?;
    writeln!(w, "element vertex {}", mesh.num_vertices()).map_err(io_err)?;
    writeln!(w, "property float x").map_err(io_err)?;
    writeln!(w, "property float y").map_err(io_err)?;
    writeln!(w, "property float z").map_err(io_err)?;
    writeln!(w, "property float nx").map_err(io_err)?;
    writeln!(w, "property float ny").map_err(io_err)?;
    writeln!(w, "property float nz").map_err(io_err)?;
    writeln!(w, "element face {}", mesh.num_faces()).map_err(io_err)?;
    writeln!(w, "property list uchar int vertex_indices").map_err(io_err)?;
    writeln!(w, "end_header").map_err(io_err)?;

    for v in &mesh.vertices {
        writeln!(
            w,
            "{} {} {} {} {} {}",
            v.position[0], v.position[1], v.position[2], v.normal[0], v.normal[1], v.normal[2],
        )
        .map_err(io_err)?;
    }

    for tri in &mesh.indices {
        writeln!(w, "3 {} {} {}", tri[0], tri[1], tri[2]).map_err(io_err)?;
    }

    w.flush().map_err(io_err)?;
    Ok(())
}

fn io_err(e: std::io::Error) -> Error {
    Error::Gpu(format!("I/O error: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types_3d::{Point3D, Vertex3D};
    use std::fs;

    #[test]
    fn ply_ascii_roundtrip() {
        let cloud = PointCloud {
            points: vec![
                Point3D {
                    position: [1.0, 2.0, 3.0],
                    color: [255, 0, 0, 255],
                    normal: [0.0; 3],
                },
                Point3D {
                    position: [4.0, 5.0, 6.0],
                    color: [0, 255, 0, 255],
                    normal: [0.0; 3],
                },
            ],
        };

        let dir = std::env::temp_dir().join("vx_test_ply_ascii");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("test.ply");

        write_ply_ascii(&path, &cloud).unwrap();

        let content = fs::read_to_string(&path).unwrap();
        assert!(content.starts_with("ply\n"));
        assert!(content.contains("element vertex 2"));
        assert!(content.contains("1 2 3 255 0 0 255"));
        assert!(content.contains("4 5 6 0 255 0 255"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn ply_binary_writes() {
        let cloud = PointCloud {
            points: vec![Point3D {
                position: [1.0, 2.0, 3.0],
                color: [255, 128, 64, 255],
                normal: [0.0; 3],
            }],
        };

        let dir = std::env::temp_dir().join("vx_test_ply_binary");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("test.ply");

        write_ply_binary(&path, &cloud).unwrap();

        let data = fs::read(&path).unwrap();
        // Should start with "ply\n"
        assert_eq!(&data[..4], b"ply\n");
        // Should contain binary marker
        let header = String::from_utf8_lossy(&data);
        assert!(header.contains("binary_little_endian"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn obj_roundtrip() {
        let mesh = Mesh {
            vertices: vec![
                Vertex3D {
                    position: [0.0, 0.0, 0.0],
                    normal: [0.0, 0.0, 1.0],
                    uv: [0.0; 2],
                },
                Vertex3D {
                    position: [1.0, 0.0, 0.0],
                    normal: [0.0, 0.0, 1.0],
                    uv: [0.0; 2],
                },
                Vertex3D {
                    position: [0.0, 1.0, 0.0],
                    normal: [0.0, 0.0, 1.0],
                    uv: [0.0; 2],
                },
            ],
            indices: vec![[0, 1, 2]],
        };

        let dir = std::env::temp_dir().join("vx_test_obj");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("test.obj");

        write_obj(&path, &mesh).unwrap();

        let content = fs::read_to_string(&path).unwrap();
        assert!(content.contains("v 0 0 0"));
        assert!(content.contains("v 1 0 0"));
        assert!(content.contains("vn 0 0 1"));
        // OBJ uses 1-based indices
        assert!(content.contains("f 1//1 2//2 3//3"));

        let _ = fs::remove_dir_all(&dir);
    }
}
