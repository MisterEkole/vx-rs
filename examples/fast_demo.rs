// examples/fast_demo.rs
//
// End-to-end: FAST corner detection → Harris scoring → top-10 by Harris response.
//
// Run with:
//   cargo run --example fast_demo -- path/to/image.png

use vx_vision::Context;
use vx_vision::kernels::fast::{FastDetectConfig, FastDetector};
use vx_vision::kernels::harris::{HarrisConfig, HarrisScorer};

fn main() {
    // --- One-liner GPU setup ---
    let ctx = Context::new().expect("No Metal GPU available");

    // --- Build pipelines once ---
    let fast = FastDetector::new(&ctx).expect("Failed to create FAST pipeline");
    let harris = HarrisScorer::new(&ctx).expect("Failed to create Harris pipeline");

    // --- Load image ---
    let path = std::env::args()
        .nth(1)
        .expect("Usage: fast_demo <image_path>");
    let img = image::open(&path)
        .expect("Failed to open image")
        .to_luma8();
    let (w, h) = img.dimensions();

    // --- Upload to GPU texture ---
    let texture = ctx
        .texture_gray8(img.as_raw(), w, h)
        .expect("Failed to create texture");

    // --- Stage 1: FAST detection ---
    let fast_config = FastDetectConfig {
        threshold: 20,
        max_corners: 4096,
    };
    let fast_result = fast
        .detect(&ctx, &texture, &fast_config)
        .expect("FAST detection failed");

    println!(
        "FAST: {} candidate corners in {}x{} image",
        fast_result.corners.len(), w, h
    );

    // --- Stage 2: Harris scoring ---
    // Takes the FAST corners + the same image texture, replaces each
    // corner's response with the Harris measure R = det(M) - k·trace(M)²
    let harris_config = HarrisConfig {
        k: 0.04,
        patch_radius: 3,   // 7×7 integration window
    };
    let mut scored = harris
        .compute(&ctx, &texture, &fast_result.corners, &harris_config)
        .expect("Harris scoring failed");

    // --- Sort by Harris response (descending) and print top-10 ---
    scored.sort_by(|a, b| b.response.partial_cmp(&a.response).unwrap());

    println!("\nTop 10 corners by Harris response:");
    for (i, c) in scored.iter().take(10).enumerate() {
        println!(
            "  #{:>2}  ({:>6.1}, {:>6.1})  harris={:.4}",
            i + 1,
            c.position[0],
            c.position[1],
            c.response,
        );
    }
}