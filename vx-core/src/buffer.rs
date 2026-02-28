// vx-core/src/buffer.rs
//
// UnifiedBuffer<T>: A typed, shared-memory Metal buffer.
//
// On Apple Silicon (UMA), MTLStorageModeShared means the CPU and GPU
// share the same physical memory — zero copies.  This wrapper adds
// type safety and a GpuGuard mechanism that prevents &mut CPU access
// while the buffer is in-flight on the GPU.

use std::marker::PhantomData;
use std::mem;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use bytemuck::Pod;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

/// A typed, shared-memory Metal buffer backed by `MTLStorageModeShared`.
///
/// `T` must be `Pod` (plain old data) so it can be safely blitted
/// between CPU and GPU without serialization.
pub struct UnifiedBuffer<T: Pod> {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    count: usize,
    /// True while the buffer is submitted to a GPU command.
    /// Prevents mutable CPU access until the GPU signals completion.
    in_flight: Arc<AtomicBool>,
    _marker: PhantomData<T>,
}

impl<T: Pod> UnifiedBuffer<T> {
    /// Allocate a new shared buffer for `count` elements of `T`.
    pub fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        count: usize,
    ) -> Result<Self, String> {
        let size = mem::size_of::<T>() * count;
        let raw =device.newBufferWithLength_options(
                size,
                MTLResourceOptions::StorageModeShared,
            )
        
        .ok_or_else(|| format!("Failed to allocate {} byte shared buffer", size))?;

        Ok(Self {
            raw,
            count,
            in_flight: Arc::new(AtomicBool::new(false)),
            _marker: PhantomData,
        })
    }

    /// Number of `T` elements this buffer can hold.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Size in bytes.
    pub fn byte_size(&self) -> usize {
        mem::size_of::<T>() * self.count
    }

    /// Get a reference to the underlying Metal buffer (for binding to encoders).
    pub fn metal_buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.raw
    }

    /// Read-only slice of the buffer contents.
    ///
    /// Safe to call even while the buffer is in-flight (the GPU may be
    /// writing, so the data may be stale — but it won't crash).
    pub fn as_slice(&self) -> &[T] {
        let ptr = self.raw.contents().as_ptr() as *const T;
        unsafe { std::slice::from_raw_parts(ptr, self.count) }
    }

    /// Mutable slice of the buffer contents.
    ///
    /// # Panics
    /// Panics if the buffer is currently in-flight on the GPU.
    /// Wait for the command buffer to complete before calling this.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        assert!(
            !self.in_flight.load(Ordering::Acquire),
            "Cannot mutably access UnifiedBuffer while it is in-flight on the GPU"
        );
        let ptr = self.raw.contents().as_ptr() as *mut T;
        unsafe { std::slice::from_raw_parts_mut(ptr, self.count) }
    }

    /// Copy data from a slice into the buffer.
    ///
    /// # Panics
    /// Panics if `data.len() > self.count` or if the buffer is in-flight.
    pub fn write(&mut self, data: &[T]) {
        assert!(data.len() <= self.count, "Data exceeds buffer capacity");
        self.as_mut_slice()[..data.len()].copy_from_slice(data);
    }

    /// Copy the buffer contents to a new Vec.
    pub fn to_vec(&self) -> Vec<T> {
        self.as_slice().to_vec()
    }

    /// Acquire a GPU guard, marking this buffer as in-flight.
    /// Returns a `GpuGuard` that automatically releases on drop.
    pub fn gpu_guard(&self) -> GpuGuard<T> {
        self.in_flight.store(true, Ordering::Release);
        GpuGuard {
            in_flight: Arc::clone(&self.in_flight),
            _marker: PhantomData,
        }
    }

    /// Check if the buffer is currently in-flight on the GPU.
    pub fn is_in_flight(&self) -> bool {
        self.in_flight.load(Ordering::Acquire)
    }
}

/// RAII guard that prevents mutable CPU access to a `UnifiedBuffer`
/// while it is submitted to the GPU.
///
/// Drop the guard (or call [`GpuGuard::release`]) once the command
/// buffer has completed to re-enable `as_mut_slice()` / `write()`.
pub struct GpuGuard<T: Pod> {
    in_flight: Arc<AtomicBool>,
    _marker: PhantomData<T>,
}

impl<T: Pod> GpuGuard<T> {
    /// Explicitly release the guard (same as dropping it).
    pub fn release(self) {
        // Drop impl handles it
    }
}

impl<T: Pod> Drop for GpuGuard<T> {
    fn drop(&mut self) {
        self.in_flight.store(false, Ordering::Release);
    }
}