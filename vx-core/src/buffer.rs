//! Typed shared-memory Metal buffers with GPU in-flight guards.

use std::marker::PhantomData;
use std::mem;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use bytemuck::Pod;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

/// A typed Metal buffer using `MTLStorageModeShared` for zero-copy CPU/GPU access.
///
/// `T` must implement [`Pod`] to guarantee safe bitwise transfer between
/// CPU and GPU address spaces. Mutable CPU access is blocked while the
/// buffer is in-flight on the GPU via [`GpuGuard`].
pub struct UnifiedBuffer<T: Pod> {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    count: usize,
    in_flight: Arc<AtomicBool>,
    _marker: PhantomData<T>,
}

impl<T: Pod> UnifiedBuffer<T> {
    /// Allocates a shared buffer for `count` elements of `T`.
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

    /// Returns the number of `T` elements in this buffer.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Returns the total size in bytes.
    pub fn byte_size(&self) -> usize {
        mem::size_of::<T>() * self.count
    }

    /// Returns a reference to the underlying `MTLBuffer` for encoder binding.
    pub fn metal_buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.raw
    }

    /// Returns a read-only slice of the buffer contents.
    ///
    /// Safe to call while in-flight, though data may be stale if the
    /// GPU is actively writing.
    pub fn as_slice(&self) -> &[T] {
        let ptr = self.raw.contents().as_ptr() as *const T;
        unsafe { std::slice::from_raw_parts(ptr, self.count) }
    }

    /// Returns a mutable slice of the buffer contents.
    ///
    /// # Panics
    ///
    /// Panics if the buffer is currently in-flight on the GPU.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        assert!(
            !self.in_flight.load(Ordering::Acquire),
            "Cannot mutably access UnifiedBuffer while it is in-flight on the GPU"
        );
        let ptr = self.raw.contents().as_ptr() as *mut T;
        unsafe { std::slice::from_raw_parts_mut(ptr, self.count) }
    }

    /// Copies `data` into the buffer.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() > self.count()` or if the buffer is in-flight.
    pub fn write(&mut self, data: &[T]) {
        assert!(data.len() <= self.count, "Data exceeds buffer capacity");
        self.as_mut_slice()[..data.len()].copy_from_slice(data);
    }

    /// Copies the buffer contents into a new `Vec<T>`.
    pub fn to_vec(&self) -> Vec<T> {
        self.as_slice().to_vec()
    }

    /// Marks this buffer as in-flight and returns a [`GpuGuard`] that
    /// clears the flag on drop.
    pub fn gpu_guard(&self) -> GpuGuard<T> {
        self.in_flight.store(true, Ordering::Release);
        GpuGuard {
            in_flight: Arc::clone(&self.in_flight),
            _marker: PhantomData,
        }
    }

    /// Returns `true` if the buffer is currently in-flight on the GPU.
    pub fn is_in_flight(&self) -> bool {
        self.in_flight.load(Ordering::Acquire)
    }
}

/// RAII guard that blocks mutable CPU access to a [`UnifiedBuffer`]
/// until dropped.
pub struct GpuGuard<T: Pod> {
    in_flight: Arc<AtomicBool>,
    _marker: PhantomData<T>,
}

impl<T: Pod> GpuGuard<T> {
    /// Explicitly releases the guard. Equivalent to `drop(guard)`.
    pub fn release(self) {
        // Drop impl handles it
    }
}

impl<T: Pod> Drop for GpuGuard<T> {
    fn drop(&mut self) {
        self.in_flight.store(false, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::default_device;

    fn get_device() -> Retained<ProtocolObject<dyn objc2_metal::MTLDevice>> {
        default_device().expect("No Metal device — tests require Apple Silicon or compatible GPU")
    }

    #[test]
    fn create_buffer() {
        let device = get_device();
        let buf = UnifiedBuffer::<f32>::new(&device, 1024).expect("Failed to allocate buffer");
        assert_eq!(buf.count(), 1024);
        assert_eq!(buf.byte_size(), 1024 * 4);
    }

    #[test]
    fn read_write_roundtrip() {
        let device = get_device();
        let mut buf = UnifiedBuffer::<u32>::new(&device, 4).expect("alloc");
        buf.write(&[10, 20, 30, 40]);
        assert_eq!(buf.as_slice(), &[10, 20, 30, 40]);
    }

    #[test]
    fn to_vec() {
        let device = get_device();
        let mut buf = UnifiedBuffer::<i32>::new(&device, 3).expect("alloc");
        buf.write(&[1, 2, 3]);
        assert_eq!(buf.to_vec(), vec![1, 2, 3]);
    }

    #[test]
    fn as_mut_slice_direct() {
        let device = get_device();
        let mut buf = UnifiedBuffer::<u8>::new(&device, 4).expect("alloc");
        let slice = buf.as_mut_slice();
        slice[0] = 0xAA;
        slice[1] = 0xBB;
        assert_eq!(buf.as_slice()[0], 0xAA);
        assert_eq!(buf.as_slice()[1], 0xBB);
    }

    #[test]
    fn gpu_guard_prevents_mutation() {
        let device = get_device();
        let buf = UnifiedBuffer::<f32>::new(&device, 4).expect("alloc");
        assert!(!buf.is_in_flight());

        let guard = buf.gpu_guard();
        assert!(buf.is_in_flight());

        drop(guard);
        assert!(!buf.is_in_flight());
    }

    #[test]
    #[should_panic(expected = "Cannot mutably access")]
    fn mut_access_panics_while_in_flight() {
        let device = get_device();
        let mut buf = UnifiedBuffer::<f32>::new(&device, 4).expect("alloc");
        let _guard = buf.gpu_guard();
        let _ = buf.as_mut_slice(); // should panic
    }

    #[test]
    fn zero_initialized() {
        let device = get_device();
        let buf = UnifiedBuffer::<u32>::new(&device, 8).expect("alloc");
        // Metal shared buffers are zero-initialized
        for &val in buf.as_slice() {
            assert_eq!(val, 0);
        }
    }

    #[test]
    fn large_buffer() {
        let device = get_device();
        let n = 1_000_000;
        let mut buf = UnifiedBuffer::<f32>::new(&device, n).expect("alloc");
        assert_eq!(buf.count(), n);
        buf.as_mut_slice()[n - 1] = 42.0;
        assert_eq!(buf.as_slice()[n - 1], 42.0);
    }
}