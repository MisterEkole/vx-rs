// examples/fast_demo.rs
//
// FAST corner detection -> Harris scoring -> top-10 by Harris response.
//
// Run with:
//   cargo run --example fast_demo -- path/to/image.png

use vx_vision::kernels::fast::{FastDetectConfig, FastDetector};
use vx_vision::kernels::harris::{HarrisConfig, HarrisScorer};
use vx_vision::Context;

fn main() {
    let ctx = Context::new().expect("No Metal GPU available");
    let fast = FastDetector::new(&ctx).expect("Failed to create FAST pipeline");
    let harris = HarrisScorer::new(&ctx).expect("Failed to create Harris pipeline");

    let path = std::env::args()
        .nth(1)
        .expect("Usage: fast_demo <image_path>");
    let img = image::open(&path).expect("Failed to open image").to_luma8();
    let (w, h) = img.dimensions();

    let texture = ctx
        .texture_gray8(img.as_raw(), w, h)
        .expect("Failed to create texture");

    let fast_config = FastDetectConfig::new(20, 4096);
    let fast_result = fast
        .detect(&ctx, &texture, &fast_config)
        .expect("FAST detection failed");

    println!(
        "FAST: {} candidate corners in {}x{} image",
        fast_result.corners.len(),
        w,
        h
    );

    // Harris replaces each corner's response with R = det(M) - k*trace(M)^2
    let harris_config = HarrisConfig::new(0.04, 3);
    let mut scored = harris
        .compute(&ctx, &texture, &fast_result.corners, &harris_config)
        .expect("Harris scoring failed");

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
