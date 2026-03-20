// Integration tests for TexturePool

use vx_vision::{Context, TexturePool};

#[test]
fn pool_acquire_and_release() {
    let ctx = Context::new().unwrap();
    let mut pool = TexturePool::new();

    // First acquire = allocation (miss)
    let tex = pool.acquire_gray8(&ctx, 64, 64).unwrap();
    assert_eq!(tex.width(), 64);
    assert_eq!(tex.height(), 64);
    assert_eq!(pool.total_acquires(), 1);
    assert_eq!(pool.total_hits(), 0);

    // Release back to pool
    pool.release(tex);
    assert_eq!(pool.cached_count(), 1);

    // Second acquire = reuse (hit)
    let tex2 = pool.acquire_gray8(&ctx, 64, 64).unwrap();
    assert_eq!(pool.total_acquires(), 2);
    assert_eq!(pool.total_hits(), 1);
    assert_eq!(tex2.width(), 64);
}

#[test]
fn pool_different_sizes_no_hit() {
    let ctx = Context::new().unwrap();
    let mut pool = TexturePool::new();

    let tex = pool.acquire_gray8(&ctx, 64, 64).unwrap();
    pool.release(tex);

    // Different size = miss
    let _tex2 = pool.acquire_gray8(&ctx, 32, 32).unwrap();
    assert_eq!(pool.total_hits(), 0);
}

#[test]
fn pool_different_formats() {
    let ctx = Context::new().unwrap();
    let mut pool = TexturePool::new();

    let t1 = pool.acquire_r32float(&ctx, 64, 64).unwrap();
    let t2 = pool.acquire_rgba8(&ctx, 64, 64).unwrap();
    let t3 = pool.acquire_gray8(&ctx, 64, 64).unwrap();

    assert_eq!(pool.total_acquires(), 3);
    assert_eq!(pool.total_hits(), 0);

    pool.release(t1);
    pool.release(t2);
    pool.release(t3);
    assert_eq!(pool.cached_count(), 3);
}

#[test]
fn pool_clear() {
    let ctx = Context::new().unwrap();
    let mut pool = TexturePool::new();

    let tex = pool.acquire_gray8(&ctx, 64, 64).unwrap();
    pool.release(tex);
    assert_eq!(pool.cached_count(), 1);

    pool.clear();
    assert_eq!(pool.cached_count(), 0);
}

#[test]
fn pool_hit_rate() {
    let ctx = Context::new().unwrap();
    let mut pool = TexturePool::new();

    // 0 acquires = 0 hit rate
    assert_eq!(pool.hit_rate(), 0.0);

    let tex = pool.acquire_gray8(&ctx, 64, 64).unwrap();
    pool.release(tex);
    let tex = pool.acquire_gray8(&ctx, 64, 64).unwrap();
    pool.release(tex);

    // 2 acquires, 1 hit = 50%
    assert!((pool.hit_rate() - 0.5).abs() < 0.01);
}

#[test]
fn pool_capacity_limit() {
    let ctx = Context::new().unwrap();
    let mut pool = TexturePool::with_capacity(2);

    // Release 3 textures, but capacity is 2
    let t1 = pool.acquire_gray8(&ctx, 64, 64).unwrap();
    let t2 = pool.acquire_gray8(&ctx, 64, 64).unwrap();
    let t3 = pool.acquire_gray8(&ctx, 64, 64).unwrap();

    pool.release(t1);
    pool.release(t2);
    pool.release(t3); // should be dropped, exceeds capacity

    assert_eq!(pool.cached_count(), 2);
}
