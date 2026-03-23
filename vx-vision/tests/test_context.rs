// Integration tests for Context and Texture

use vx_vision::{Context, TextureFormat};

#[test]
fn context_creation() {
    let ctx = Context::new();
    assert!(ctx.is_ok(), "Context::new() failed: {:?}", ctx.err());
}

#[test]
fn texture_gray8_roundtrip() {
    let ctx = Context::new().unwrap();
    let w = 16;
    let h = 16;
    let pixels: Vec<u8> = (0..w * h).map(|i| (i % 256) as u8).collect();

    let tex = ctx.texture_gray8(&pixels, w as u32, h as u32).unwrap();
    assert_eq!(tex.width(), w as u32);
    assert_eq!(tex.height(), h as u32);
    assert_eq!(tex.format(), TextureFormat::R8Unorm);

    let readback = tex.read_gray8();
    assert_eq!(readback.len(), pixels.len());
    assert_eq!(readback, pixels);
}

#[test]
fn texture_rgba8_roundtrip() {
    let ctx = Context::new().unwrap();
    let w = 8u32;
    let h = 8u32;
    let pixels: Vec<u8> = (0..(w * h * 4) as usize).map(|i| (i % 256) as u8).collect();

    let tex = ctx.texture_rgba8(&pixels, w, h).unwrap();
    assert_eq!(tex.format(), TextureFormat::RGBA8Unorm);

    let readback = tex.read_rgba8();
    assert_eq!(readback, pixels);
}

#[test]
fn texture_r32float_roundtrip() {
    let ctx = Context::new().unwrap();
    let w = 4u32;
    let h = 4u32;
    let data: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();

    let tex = ctx.texture_r32float(&data, w, h).unwrap();
    assert_eq!(tex.format(), TextureFormat::R32Float);

    let readback = tex.read_r32float();
    assert_eq!(readback.len(), 16);
    for (a, b) in readback.iter().zip(data.iter()) {
        assert!((a - b).abs() < 1e-6, "Mismatch: {} vs {}", a, b);
    }
}

#[test]
fn texture_output_gray8() {
    let ctx = Context::new().unwrap();
    let tex = ctx.texture_output_gray8(64, 64).unwrap();
    assert_eq!(tex.width(), 64);
    assert_eq!(tex.height(), 64);
}

#[test]
fn texture_output_r32float() {
    let ctx = Context::new().unwrap();
    let tex = ctx.texture_output_r32float(32, 32).unwrap();
    assert_eq!(tex.width(), 32);
    assert_eq!(tex.height(), 32);
    assert_eq!(tex.format(), TextureFormat::R32Float);
}

#[test]
fn texture_wrong_size_rejected() {
    let ctx = Context::new().unwrap();
    let result = ctx.texture_gray8(&[0u8; 10], 4, 4); // needs 16 bytes
    assert!(result.is_err());
}
