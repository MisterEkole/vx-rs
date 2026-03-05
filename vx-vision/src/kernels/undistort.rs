// vx-vision/src/kernels/undistort.rs
//
// Rust binding for the lens undistortion kernel (undistort.metal).
//
// The shader remaps every output pixel by sampling the input image at the
// (x, y) coordinate stored in two precomputed float maps (map_x, map_y).
// This is the standard OpenCV-style remap: given camera intrinsics and
// distortion coefficients you compute the maps once on the CPU, upload them
// as R32Float textures, then call this kernel every frame.
//
// Usage:
//   let ctx = Context::new()?;
//   let undistorter = Undistorter::new(&ctx)?;
//
//   // Build maps once (e.g. from OpenCV initUndistortRectifyMap output):
//   let map_x = ctx.texture_r32float(&map_x_data, width, height)?;
//   let map_y = ctx.texture_r32float(&map_y_data, width, height)?;
//   let output = ctx.texture_output_gray8(width, height)?;
//
//   // Per frame:
//   undistorter.apply(&ctx, &input, &map_x, &map_y, &output)?;
//   let pixels: Vec<u8> = output.read_gray8();

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice,
    MTLLibrary, MTLSize,
};

use crate::context::Context;
use crate::texture::Texture;

// ---------------------------------------------------------------------------
// Undistorter
// ---------------------------------------------------------------------------

/// Compiled lens undistortion pipeline. Create once, reuse across frames.
///
/// The remap maps (`map_x`, `map_y`) are typically computed once from camera
/// calibration data and reused for every frame.
pub struct Undistorter {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl Undistorter {
    // --------------------------------------------------------------------
    // Construction
    // --------------------------------------------------------------------

    /// Build the compute pipeline from the context's shader library.
    pub fn new(ctx: &Context) -> Result<Self, String> {
        let name = objc2_foundation::ns_string!("undistort");

        let func = ctx.library().newFunctionWithName(name)
            .ok_or_else(|| "Missing kernel function 'undistort'".to_string())?;

        let pipeline = ctx.device().newComputePipelineStateWithFunction_error(&func)
            .map_err(|e| format!("Failed to create undistort pipeline: {e}"))?;

        Ok(Self { pipeline })
    }

    // --------------------------------------------------------------------
    // Synchronous dispatch
    // --------------------------------------------------------------------

    /// Remap `input` into `output` using per-pixel float coordinate maps.
    ///
    /// - `input`  — distorted source image (R8Unorm, ShaderRead)
    /// - `map_x`  — per-pixel source x coordinate (R32Float, ShaderRead);
    ///              create with [`Context::texture_r32float`]
    /// - `map_y`  — per-pixel source y coordinate (R32Float, ShaderRead);
    ///              create with [`Context::texture_r32float`]
    /// - `output` — undistorted destination (R8Unorm, ShaderWrite);
    ///              create with [`Context::texture_output_gray8`], read back
    ///              with [`Texture::read_gray8`]
    ///
    /// `map_x`, `map_y`, and `output` must all be the same dimensions.
    ///
    /// Synchronous: encodes, commits, waits for GPU completion.
    pub fn apply(
        &self,
        ctx: &Context,
        input:  &Texture,
        map_x:  &Texture,
        map_y:  &Texture,
        output: &Texture,
    ) -> Result<(), String> {
        let width  = output.width();
        let height = output.height();

        let cmd_buf = ctx.queue().commandBuffer()
            .ok_or("Failed to create command buffer")?;
        let encoder = cmd_buf.computeCommandEncoder()
            .ok_or("Failed to create compute encoder")?;

        Self::encode_into(&self.pipeline, &encoder, input, map_x, map_y, output, width, height);

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        Ok(())
    }

    // --------------------------------------------------------------------
    // Encode-only (for pipelining)
    // --------------------------------------------------------------------

    /// Encode the undistortion into an existing compute encoder without
    /// committing. Useful when chaining with other kernels.
    pub fn encode(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        input:   &Texture,
        map_x:   &Texture,
        map_y:   &Texture,
        output:  &Texture,
    ) {
        let width  = output.width();
        let height = output.height();
        Self::encode_into(&self.pipeline, encoder, input, map_x, map_y, output, width, height);
    }

    // --------------------------------------------------------------------
    // Internal
    // --------------------------------------------------------------------

    fn encode_into(
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        encoder:  &ProtocolObject<dyn MTLComputeCommandEncoder>,
        input:    &Texture,
        map_x:    &Texture,
        map_y:    &Texture,
        output:   &Texture,
        width:    u32,
        height:   u32,
    ) {
        // SAFETY: GPU encoder operations interact with device state.
        //
        // Binding matches undistort.metal:
        //   texture(0) = input   (sample access — distorted source)
        //   texture(1) = map_x   (read access  — source x coords, R32Float)
        //   texture(2) = map_y   (read access  — source y coords, R32Float)
        //   texture(3) = output  (write access — undistorted destination)
        //
        // No params buffer needed; everything is encoded in the maps.
        // 2D dispatch: one thread per output pixel.
        unsafe {
            encoder.setComputePipelineState(pipeline);
            encoder.setTexture_atIndex(Some(input.raw()),  0);
            encoder.setTexture_atIndex(Some(map_x.raw()),  1);
            encoder.setTexture_atIndex(Some(map_y.raw()),  2);
            encoder.setTexture_atIndex(Some(output.raw()), 3);

            let tew    = pipeline.threadExecutionWidth();
            let max_tg = pipeline.maxTotalThreadsPerThreadgroup();
            let tg_h   = (max_tg / tew).max(1);

            let grid    = MTLSize { width: width as usize,  height: height as usize, depth: 1 };
            let tg_size = MTLSize { width: tew,             height: tg_h,            depth: 1 };

            encoder.dispatchThreads_threadsPerThreadgroup(grid, tg_size);
        }
    }
}
