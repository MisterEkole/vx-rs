//! Criterion benchmarks for core GPU kernels.
//!
//! Run with: cargo bench -p vx-vision

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use vx_vision::Context;
use vx_vision::kernels::fast::{FastDetector, FastDetectConfig};
use vx_vision::kernels::harris::{HarrisScorer, HarrisConfig};
use vx_vision::kernels::nms::{NmsSuppressor, NmsConfig};
use vx_vision::kernels::orb::{OrbDescriptor, OrbConfig};
use vx_vision::kernels::gaussian::{GaussianBlur, GaussianConfig};

/// Generates a synthetic gradient + noise image for benchmarking.
fn make_bench_image(w: u32, h: u32) -> Vec<u8> {
    (0..(w * h)).map(|i| {
        let x = i % w;
        let y = i / w;
        let grad = ((x as f32 / w as f32) * 255.0) as u8;
        let noise = ((x.wrapping_mul(17) ^ y.wrapping_mul(31)) % 64) as u8;
        grad.wrapping_add(noise)
    }).collect()
}

/// Standard ORB test pattern (256 pairs x 4 offsets = 1024 values).
fn orb_pattern() -> Vec<i32> {
    let mut pattern = Vec::with_capacity(1024);
    for i in 0..256u32 {
        let angle = (i as f32) * std::f32::consts::TAU / 256.0;
        let r1 = 5.0 + (i as f32 % 8.0);
        let r2 = 5.0 + ((i + 128) as f32 % 8.0);
        pattern.push((r1 * angle.cos()) as i32);
        pattern.push((r1 * angle.sin()) as i32);
        pattern.push((r2 * (angle + 0.5).cos()) as i32);
        pattern.push((r2 * (angle + 0.5).sin()) as i32);
    }
    pattern
}

fn bench_fast(c: &mut Criterion) {
    let ctx = Context::new().unwrap();
    let det = FastDetector::new(&ctx).unwrap();
    let cfg = FastDetectConfig::default();

    let mut group = c.benchmark_group("FAST");
    for &(w, h) in &[(752, 480), (1920, 1080)] {
        let pixels = make_bench_image(w, h);
        let texture = ctx.texture_gray8(&pixels, w, h).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(format!("{w}x{h}")), &(), |b, _| {
            b.iter(|| {
                det.detect(&ctx, &texture, &cfg).unwrap()
            });
        });
    }
    group.finish();
}

fn bench_fast_harris_nms_orb(c: &mut Criterion) {
    let ctx = Context::new().unwrap();
    let fast = FastDetector::new(&ctx).unwrap();
    let harris = HarrisScorer::new(&ctx).unwrap();
    let nms = NmsSuppressor::new(&ctx).unwrap();
    let orb = OrbDescriptor::new(&ctx).unwrap();

    let fast_cfg = FastDetectConfig::default();
    let harris_cfg = HarrisConfig::default();
    let nms_cfg = NmsConfig::default();
    let orb_cfg = OrbConfig::default();
    let pattern = orb_pattern();

    let mut group = c.benchmark_group("FAST+Harris+NMS+ORB");
    for &(w, h) in &[(752, 480), (1920, 1080)] {
        let pixels = make_bench_image(w, h);
        let texture = ctx.texture_gray8(&pixels, w, h).unwrap();

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{w}x{h}")),
            &(),
            |b, _| {
                b.iter(|| {
                    let fast_result = fast.detect(&ctx, &texture, &fast_cfg).unwrap();
                    let scored = harris.compute(&ctx, &texture, &fast_result.corners, &harris_cfg).unwrap();
                    let suppressed = nms.run(&ctx, &scored, &nms_cfg).unwrap();
                    let _orb_result = orb.compute(&ctx, &texture, &suppressed, &pattern, &orb_cfg).unwrap();
                });
            },
        );
    }
    group.finish();
}

fn bench_pipeline_vs_individual(c: &mut Criterion) {
    let ctx = Context::new().unwrap();
    let gauss = GaussianBlur::new(&ctx).unwrap();

    let w = 1920u32;
    let h = 1080u32;
    let pixels = make_bench_image(w, h);
    let input = ctx.texture_gray8(&pixels, w, h).unwrap();

    let mut gauss_cfg = GaussianConfig::default();
    gauss_cfg.sigma = 1.5;
    gauss_cfg.radius = 4;

    let mut group = c.benchmark_group("Pipeline vs Individual");

    // Individual: 3 separate command buffers
    group.bench_function("3x Gaussian (individual)", |b| {
        let inter1 = ctx.texture_intermediate_gray8(w, h).unwrap();
        let inter2 = ctx.texture_intermediate_gray8(w, h).unwrap();
        let output = ctx.texture_output_gray8(w, h).unwrap();
        b.iter(|| {
            gauss.apply(&ctx, &input, &inter1, &gauss_cfg).unwrap();
            gauss.apply(&ctx, &inter1, &inter2, &gauss_cfg).unwrap();
            gauss.apply(&ctx, &inter2, &output, &gauss_cfg).unwrap();
        });
    });

    // Pipeline: single command buffer
    group.bench_function("3x Gaussian (pipeline)", |b| {
        let inter1 = ctx.texture_intermediate_gray8(w, h).unwrap();
        let inter2 = ctx.texture_intermediate_gray8(w, h).unwrap();
        let output = ctx.texture_output_gray8(w, h).unwrap();
        b.iter(|| {
            let pipe = vx_vision::Pipeline::begin(&ctx).unwrap();
            let cmd = pipe.cmd_buf();
            let s1 = gauss.encode(&ctx, cmd, &input, &inter1, &gauss_cfg).unwrap();
            let s2 = gauss.encode(&ctx, cmd, &inter1, &inter2, &gauss_cfg).unwrap();
            let s3 = gauss.encode(&ctx, cmd, &inter2, &output, &gauss_cfg).unwrap();
            let _ = (s1, s2, s3);
            let _retained = pipe.commit_and_wait();
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_fast,
    bench_fast_harris_nms_orb,
    bench_pipeline_vs_individual,
);
criterion_main!(benches);
