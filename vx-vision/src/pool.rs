//! Texture recycling pool keyed by `(width, height, format)`.

use std::collections::HashMap;

use crate::context::Context;
use crate::texture::{Texture, TextureFormat};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PoolKey {
    width:  u32,
    height: u32,
    format: TextureFormat,
}

/// Caches and reuses GPU textures to avoid repeated allocation.
pub struct TexturePool {
    buckets: HashMap<PoolKey, Vec<Texture>>,
    max_per_bucket: usize,
    acquires: u64,
    hits: u64,
}

impl TexturePool {
    /// Creates an empty pool with the default per-bucket limit (8).
    pub fn new() -> Self {
        Self {
            buckets: HashMap::new(),
            max_per_bucket: 8,
            acquires: 0,
            hits: 0,
        }
    }

    /// Creates a pool that caches at most `max_per_bucket` textures per key.
    pub fn with_capacity(max_per_bucket: usize) -> Self {
        Self {
            buckets: HashMap::new(),
            max_per_bucket,
            acquires: 0,
            hits: 0,
        }
    }

    /// Acquires an R8Unorm texture, reusing a cached one if available.
    pub fn acquire_gray8(
        &mut self,
        ctx:    &Context,
        width:  u32,
        height: u32,
    ) -> Result<Texture, String> {
        self.acquire(ctx, width, height, TextureFormat::R8Unorm)
    }

    /// Acquires an R32Float texture, reusing a cached one if available.
    pub fn acquire_r32float(
        &mut self,
        ctx:    &Context,
        width:  u32,
        height: u32,
    ) -> Result<Texture, String> {
        self.acquire(ctx, width, height, TextureFormat::R32Float)
    }

    /// Acquires an RGBA8Unorm texture, reusing a cached one if available.
    pub fn acquire_rgba8(
        &mut self,
        ctx:    &Context,
        width:  u32,
        height: u32,
    ) -> Result<Texture, String> {
        self.acquire(ctx, width, height, TextureFormat::RGBA8Unorm)
    }

    /// Acquires a texture with the given format, reusing a cached one if available.
    pub fn acquire(
        &mut self,
        ctx:    &Context,
        width:  u32,
        height: u32,
        format: TextureFormat,
    ) -> Result<Texture, String> {
        self.acquires += 1;
        let key = PoolKey { width, height, format };

        if let Some(bucket) = self.buckets.get_mut(&key) {
            if let Some(tex) = bucket.pop() {
                self.hits += 1;
                return Ok(tex);
            }
        }

        Self::create_texture(ctx, width, height, format)
    }

    /// Returns a texture to the pool for reuse. Contents are not cleared.
    pub fn release(&mut self, tex: Texture) {
        let key = PoolKey {
            width:  tex.width(),
            height: tex.height(),
            format: tex.format(),
        };

        let bucket = self.buckets.entry(key).or_default();
        if bucket.len() < self.max_per_bucket {
            bucket.push(tex);
        }
    }

    /// Returns multiple textures to the pool.
    pub fn release_all(&mut self, textures: impl IntoIterator<Item = Texture>) {
        for tex in textures {
            self.release(tex);
        }
    }

    /// Drops all cached textures.
    pub fn clear(&mut self) {
        self.buckets.clear();
    }

    /// Returns the number of textures currently cached.
    pub fn cached_count(&self) -> usize {
        self.buckets.values().map(|b| b.len()).sum()
    }

    /// Returns the total number of acquire calls.
    pub fn total_acquires(&self) -> u64 {
        self.acquires
    }

    /// Returns the number of cache hits.
    pub fn total_hits(&self) -> u64 {
        self.hits
    }

    /// Returns the cache hit rate in `[0.0, 1.0]`.
    pub fn hit_rate(&self) -> f64 {
        if self.acquires == 0 { 0.0 }
        else { self.hits as f64 / self.acquires as f64 }
    }

    fn create_texture(
        ctx: &Context,
        width: u32,
        height: u32,
        format: TextureFormat,
    ) -> Result<Texture, String> {
        match format {
            TextureFormat::R8Unorm => Texture::intermediate_gray8(ctx.device(), width, height),
            TextureFormat::R32Float => Texture::intermediate_r32float(ctx.device(), width, height),
            TextureFormat::RGBA8Unorm => Texture::output_rgba8(ctx.device(), width, height),
        }
    }
}

impl Default for TexturePool {
    fn default() -> Self {
        Self::new()
    }
}
