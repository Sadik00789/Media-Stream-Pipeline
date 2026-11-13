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
        let effective_align = alignment.max(std::mem::align_of::<T>());
        let total_bytes = capacity
            .checked_mul(std::mem::size_of::<T>())
            .expect("Capacity overflow");

        if total_bytes == 0 {
            let layout = Layout::from_size_align(0, effective_align)
                .expect("Invalid layout parameters");
            return Self {
                ptr: NonNull::dangling(),
                len: capacity,
                capacity,
                layout,
            };
        }

        let layout = Layout::from_size_align(total_bytes, effective_align)
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
        if self.layout.size() > 0 {
            unsafe {
                dealloc(self.ptr.as_ptr() as *mut u8, self.layout);
            }
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

    #[test]
    fn test_aligned_vec_edge_cases() {
        let empty: AlignedVec<f32> = AlignedVec::with_capacity_aligned(0, 64);
        assert_eq!(empty.len(), 0);
        assert_eq!(empty.capacity(), 0);
        assert!(empty.is_empty());

        let mut avx512: AlignedVec<f32> = AlignedVec::with_capacity_aligned(128, 64);
        assert!(avx512.is_aligned_to(64));
        avx512[0] = 42.5;
        assert_eq!(avx512[0], 42.5);

        let zst: AlignedVec<()> = AlignedVec::with_capacity_aligned(10, 16);
        assert_eq!(zst.len(), 10);
    }
}