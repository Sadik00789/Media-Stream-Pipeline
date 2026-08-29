use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

/// Memory-aligned vector suitable for AVX2 (32-byte) and AVX-512 (64-byte) SIMD operations.
#[derive(Debug)]
pub struct AlignedVec<T: Copy> {
    ptr: NonNull<T>,
    len: usize,
    capacity: usize,
    layout: Layout,
}

impl<T: Copy> AlignedVec<T> {
    pub fn with_capacity_aligned(capacity: usize, alignment: usize) -> Self {
        assert!(alignment.is_power_of_two());
        let layout = Layout::from_size_align(capacity * std::mem::size_of::<T>(), alignment)
            .expect("Invalid layout parameters");
        
        let raw_ptr = unsafe { alloc_zeroed(layout) as *mut T };
        let ptr = NonNull::new(raw_ptr).expect("Memory allocation failed");

        Self {
            ptr,
            len: capacity,
            capacity,
            layout,
        }
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    pub fn is_aligned_to(&self, alignment: usize) -> bool {
        (self.ptr.as_ptr() as usize) % alignment == 0
    }
}

impl<T: Copy> Deref for AlignedVec<T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T: Copy> DerefMut for AlignedVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T: Copy> Drop for AlignedVec<T> {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr.as_ptr() as *mut u8, self.layout);
        }
    }
}

unsafe impl<T: Copy + Send> Send for AlignedVec<T> {}
unsafe impl<T: Copy + Sync> Sync for AlignedVec<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_vec_32_byte_boundary() {
        let vec: AlignedVec<f32> = AlignedVec::with_capacity_aligned(512, 32);
        assert_eq!(vec.len(), 512);
        assert_eq!(vec.capacity(), 512);
        assert!(vec.is_aligned_to(32));
    }
}