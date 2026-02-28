// vx-vision/build.rs
//
// Compilation pipeline:
//   1. Collect all .metal files from shaders/
//   2. Compile each to .air with `xcrun -sdk macosx metal`
//   3. Link all .air into vx.metallib with `xcrun metallib`
//   4. Place in OUT_DIR for include_bytes!

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let shader_dir = Path::new("shaders");

    let metal_files: Vec<PathBuf> = fs::read_dir(shader_dir)
        .expect("Cannot read shaders/ directory")
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.extension().map(|e| e == "metal").unwrap_or(false) {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    if metal_files.is_empty() {
        fs::write(out_dir.join("vx.metallib"), b"").unwrap();
        println!("cargo:warning=No .metal files found in shaders/");
        return;
    }

    // Step 1: .metal → .air
    let mut air_files = Vec::new();

    for metal_path in &metal_files {
        let stem = metal_path.file_stem().unwrap().to_str().unwrap();
        let air_path = out_dir.join(format!("{stem}.air"));

        let status = Command::new("xcrun")
            .args([
                "-sdk", "macosx", "metal",
                "-c",
                "-target", "air64-apple-macos14.0",
                "-std=metal3.1",
                "-O2",
                "-Wno-unused-variable",
                metal_path.to_str().unwrap(),
                "-o", air_path.to_str().unwrap(),
            ])
            .status()
            .expect("Failed to invoke `xcrun metal` — is Xcode installed?");

        assert!(
            status.success(),
            "Shader compilation failed for {}",
            metal_path.display()
        );

        air_files.push(air_path);
        println!("cargo:rerun-if-changed={}", metal_path.display());
    }

    // Step 2: .air → vx.metallib
    let metallib_path = out_dir.join("vx.metallib");

    let mut cmd = Command::new("xcrun");
    cmd.args(["-sdk", "macosx", "metallib"]);
    for air in &air_files {
        cmd.arg(air.to_str().unwrap());
    }
    cmd.args(["-o", metallib_path.to_str().unwrap()]);

    let status = cmd.status().expect("Failed to invoke `xcrun metallib`");
    assert!(status.success(), "metallib linking failed");

    println!(
        "cargo:warning=Compiled {} shaders → {}",
        metal_files.len(),
        metallib_path.display()
    );
}