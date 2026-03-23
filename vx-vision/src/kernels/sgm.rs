//! Semi-Global Matching stereo depth estimation.

use core::ffi::c_void;
use core::ptr::NonNull;
use std::mem;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLLibrary, MTLSize,
};

use crate::context::Context;
use crate::error::{Error, Result};
use crate::texture::Texture;
use crate::types::SGMParams;
use vx_gpu::UnifiedBuffer;

#[cfg(feature = "reconstruction")]
use crate::types_3d::{CameraIntrinsics, DepthMap};

/// Configuration for SGM stereo matching.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct SGMStereoConfig {
    /// Number of disparity levels to search.
    pub num_disparities: u32,
    /// Penalty for small disparity changes (±1 pixel).
    pub p1: f32,
    /// Penalty for large disparity changes (>±1 pixel).
    pub p2: f32,
    /// Census transform horizontal half-radius.
    pub census_radius_x: u32,
    /// Census transform vertical half-radius.
    pub census_radius_y: u32,
}

impl Default for SGMStereoConfig {
    fn default() -> Self {
        Self {
            num_disparities: 128,
            p1: 10.0,
            p2: 120.0,
            census_radius_x: 4,
            census_radius_y: 3,
        }
    }
}

impl SGMStereoConfig {
    /// Creates a config with the given disparity range.
    pub fn new(num_disparities: u32) -> Self {
        Self {
            num_disparities,
            ..Default::default()
        }
    }
}

/// Compiled SGM stereo pipelines.
pub struct SGMStereo {
    census_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    aggregate_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    wta_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

unsafe impl Send for SGMStereo {}
unsafe impl Sync for SGMStereo {}

impl SGMStereo {
    /// Compiles all three SGM pipelines from the embedded metallib.
    pub fn new(ctx: &Context) -> Result<Self> {
        let census_func = ctx
            .library()
            .newFunctionWithName(objc2_foundation::ns_string!("sgm_census_transform"))
            .ok_or(Error::ShaderMissing("sgm_census_transform".into()))?;
        let aggregate_func = ctx
            .library()
            .newFunctionWithName(objc2_foundation::ns_string!("sgm_cost_aggregate"))
            .ok_or(Error::ShaderMissing("sgm_cost_aggregate".into()))?;
        let wta_func = ctx
            .library()
            .newFunctionWithName(objc2_foundation::ns_string!("sgm_wta_disparity"))
            .ok_or(Error::ShaderMissing("sgm_wta_disparity".into()))?;

        let census_pipeline = ctx
            .device()
            .newComputePipelineStateWithFunction_error(&census_func)
            .map_err(|e| Error::PipelineCompile(format!("sgm_census_transform: {e}")))?;
        let aggregate_pipeline = ctx
            .device()
            .newComputePipelineStateWithFunction_error(&aggregate_func)
            .map_err(|e| Error::PipelineCompile(format!("sgm_cost_aggregate: {e}")))?;
        let wta_pipeline = ctx
            .device()
            .newComputePipelineStateWithFunction_error(&wta_func)
            .map_err(|e| Error::PipelineCompile(format!("sgm_wta_disparity: {e}")))?;

        Ok(Self {
            census_pipeline,
            aggregate_pipeline,
            wta_pipeline,
        })
    }

    /// Computes a disparity map from rectified stereo images. Synchronous.
    pub fn compute_disparity(
        &self,
        ctx: &Context,
        left: &Texture,
        right: &Texture,
        output: &Texture,
        config: &SGMStereoConfig,
    ) -> Result<()> {
        let w = left.width();
        let h = left.height();
        let nd = config.num_disparities;
        let npixels = w as usize * h as usize;

        // Allocate intermediate buffers
        let left_census: UnifiedBuffer<[u32; 2]> = UnifiedBuffer::new(ctx.device(), npixels)?;
        let right_census: UnifiedBuffer<[u32; 2]> = UnifiedBuffer::new(ctx.device(), npixels)?;

        let cv_size = npixels * nd as usize;
        let mut cost_volume: UnifiedBuffer<u16> = UnifiedBuffer::new(ctx.device(), cv_size)?;
        // Zero the cost volume
        let zeros = vec![0u16; cv_size];
        cost_volume.write(&zeros);

        let _lc = left_census.gpu_guard();
        let _rc = right_census.gpu_guard();
        let _cv = cost_volume.gpu_guard();

        let cmd_buf = ctx
            .queue()
            .commandBuffer()
            .ok_or(Error::Gpu("failed to create command buffer".into()))?;

        {
            let encoder = cmd_buf
                .computeCommandEncoder()
                .ok_or(Error::Gpu("failed to create compute encoder".into()))?;

            let params = SGMParams {
                width: w,
                height: h,
                num_disparities: nd,
                p1: config.p1,
                p2: config.p2,
                census_radius_x: config.census_radius_x,
                census_radius_y: config.census_radius_y,
                direction_x: 0,
                direction_y: 0,
            };

            Self::encode_census(
                &self.census_pipeline,
                &encoder,
                left,
                &left_census,
                &params,
                w,
                h,
            );
            encoder.endEncoding();
        }
        {
            let encoder = cmd_buf
                .computeCommandEncoder()
                .ok_or(Error::Gpu("failed to create compute encoder".into()))?;

            let params = SGMParams {
                width: w,
                height: h,
                num_disparities: nd,
                p1: config.p1,
                p2: config.p2,
                census_radius_x: config.census_radius_x,
                census_radius_y: config.census_radius_y,
                direction_x: 0,
                direction_y: 0,
            };

            Self::encode_census(
                &self.census_pipeline,
                &encoder,
                right,
                &right_census,
                &params,
                w,
                h,
            );
            encoder.endEncoding();
        }

        let scan_dirs: [(u32, i32, u32); 2] = [
            (1, 0, h), // horizontal: one thread per row
            (0, 1, w), // vertical: one thread per column
        ];

        for &(dx, dy, scan_length) in &scan_dirs {
            let encoder = cmd_buf
                .computeCommandEncoder()
                .ok_or(Error::Gpu("failed to create compute encoder".into()))?;

            let params = SGMParams {
                width: w,
                height: h,
                num_disparities: nd,
                p1: config.p1,
                p2: config.p2,
                census_radius_x: config.census_radius_x,
                census_radius_y: config.census_radius_y,
                direction_x: dx,
                direction_y: dy,
            };

            Self::encode_aggregate(
                &self.aggregate_pipeline,
                &encoder,
                &left_census,
                &right_census,
                &cost_volume,
                &params,
                scan_length,
            );
            encoder.endEncoding();
        }

        {
            let encoder = cmd_buf
                .computeCommandEncoder()
                .ok_or(Error::Gpu("failed to create compute encoder".into()))?;

            let params = SGMParams {
                width: w,
                height: h,
                num_disparities: nd,
                p1: config.p1,
                p2: config.p2,
                census_radius_x: config.census_radius_x,
                census_radius_y: config.census_radius_y,
                direction_x: 0,
                direction_y: 0,
            };

            Self::encode_wta(
                &self.wta_pipeline,
                &encoder,
                &cost_volume,
                output,
                &params,
                w,
                h,
            );
            encoder.endEncoding();
        }

        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        drop((_lc, _rc, _cv));
        Ok(())
    }

    /// Computes a DepthMap from rectified stereo images. Synchronous.
    #[cfg(feature = "reconstruction")]
    pub fn compute(
        &self,
        ctx: &Context,
        left: &Texture,
        right: &Texture,
        config: &SGMStereoConfig,
        intrinsics: &CameraIntrinsics,
        _baseline: f32,
    ) -> Result<DepthMap> {
        let w = left.width();
        let h = left.height();
        let output = Texture::output_r32float(ctx.device(), w, h)?;

        self.compute_disparity(ctx, left, right, &output, config)?;

        // Convert disparity to depth would be done as a post-process
        // For now, the output is a disparity map. The caller can convert
        // disparity → depth via: depth = fx * baseline / disparity
        DepthMap::new(output, *intrinsics, 0.0, f32::MAX)
    }

    fn encode_census(
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        image: &Texture,
        census: &UnifiedBuffer<[u32; 2]>,
        params: &SGMParams,
        w: u32,
        h: u32,
    ) {
        unsafe {
            encoder.setComputePipelineState(pipeline);
            encoder.setTexture_atIndex(Some(image.raw()), 0);
            encoder.setBuffer_offset_atIndex(Some(census.metal_buffer()), 0, 0);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(params as *const SGMParams as *mut c_void),
                mem::size_of::<SGMParams>(),
                1,
            );

            let tew = pipeline.threadExecutionWidth();
            let max_tg = pipeline.maxTotalThreadsPerThreadgroup();
            let tg_h = (max_tg / tew).max(1);
            let grid = MTLSize {
                width: w as usize,
                height: h as usize,
                depth: 1,
            };
            let tg_size = MTLSize {
                width: tew,
                height: tg_h,
                depth: 1,
            };
            encoder.dispatchThreads_threadsPerThreadgroup(grid, tg_size);
        }
    }

    fn encode_aggregate(
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        left_census: &UnifiedBuffer<[u32; 2]>,
        right_census: &UnifiedBuffer<[u32; 2]>,
        cost_volume: &UnifiedBuffer<u16>,
        params: &SGMParams,
        scan_length: u32,
    ) {
        unsafe {
            encoder.setComputePipelineState(pipeline);
            encoder.setBuffer_offset_atIndex(Some(left_census.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(right_census.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(cost_volume.metal_buffer()), 0, 2);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(params as *const SGMParams as *mut c_void),
                mem::size_of::<SGMParams>(),
                3,
            );

            let tew = pipeline.threadExecutionWidth();
            let grid = MTLSize {
                width: scan_length as usize,
                height: 1,
                depth: 1,
            };
            let tg_size = MTLSize {
                width: tew,
                height: 1,
                depth: 1,
            };
            encoder.dispatchThreads_threadsPerThreadgroup(grid, tg_size);
        }
    }

    fn encode_wta(
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        cost_volume: &UnifiedBuffer<u16>,
        output: &Texture,
        params: &SGMParams,
        w: u32,
        h: u32,
    ) {
        unsafe {
            encoder.setComputePipelineState(pipeline);
            encoder.setBuffer_offset_atIndex(Some(cost_volume.metal_buffer()), 0, 0);
            encoder.setTexture_atIndex(Some(output.raw()), 0);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(params as *const SGMParams as *mut c_void),
                mem::size_of::<SGMParams>(),
                1,
            );

            let tew = pipeline.threadExecutionWidth();
            let max_tg = pipeline.maxTotalThreadsPerThreadgroup();
            let tg_h = (max_tg / tew).max(1);
            let grid = MTLSize {
                width: w as usize,
                height: h as usize,
                depth: 1,
            };
            let tg_size = MTLSize {
                width: tew,
                height: tg_h,
                depth: 1,
            };
            encoder.dispatchThreads_threadsPerThreadgroup(grid, tg_size);
        }
    }
}
