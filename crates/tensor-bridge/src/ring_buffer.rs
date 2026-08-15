use crate::ffi::{AudioFramePacket, SharedMemoryRingBuffer, SHM_MAGIC, SHM_RING_CAPACITY};
use std::sync::atomic::{AtomicU64, Ordering};

/// Safe zero-copy view over SharedMemoryRingBuffer
pub struct ShmRingBufferReader {
    raw: *mut SharedMemoryRingBuffer,
}

impl ShmRingBufferReader {
    /// Safety: caller must ensure `raw` points to valid initialized mmap memory.
    pub unsafe fn from_raw(raw: *mut SharedMemoryRingBuffer) -> Result<Self, &'static str> {
        if raw.is_null() {
            return Err("Shared memory pointer is null");
        }
        let magic = (*raw).magic;
        if magic != SHM_MAGIC {
            return Err("Invalid shared memory magic number");
        }
        Ok(Self { raw })
    }

    #[inline]
    pub fn write_head(&self) -> u64 {
        unsafe {
            let ptr = std::ptr::addr_of!((*self.raw).write_head) as *const AtomicU64;
            (*ptr).load(Ordering::Acquire)
        }
    }

    #[inline]
    pub fn read_head(&self) -> u64 {
        unsafe {
            let ptr = std::ptr::addr_of!((*self.raw).read_head) as *const AtomicU64;
            (*ptr).load(Ordering::Acquire)
        }
    }

    #[inline]
    pub fn advance_read_head(&mut self) -> u64 {
        unsafe {
            let ptr = std::ptr::addr_of_mut!((*self.raw).read_head) as *mut AtomicU64;
            (*ptr).fetch_add(1, Ordering::AcqRel)
        }
    }

    /// Read next available packet with automatic overrun recovery
    pub fn poll_next_packet(&mut self) -> Option<&mut AudioFramePacket> {
        let mut r = self.read_head();
        let w = self.write_head();

        if w > r + (SHM_RING_CAPACITY as u64) {
            // Overrun detected: writer wrapped around before reader caught up
            let new_r = w - (SHM_RING_CAPACITY as u64);
            let dropped = new_r - r;
            unsafe {
                let drop_ptr = std::ptr::addr_of_mut!((*self.raw).dropped_frames) as *mut AtomicU64;
                (*drop_ptr).fetch_add(dropped, Ordering::Relaxed);

                let overrun_ptr = std::ptr::addr_of_mut!((*self.raw).overrun_count) as *mut AtomicU64;
                (*overrun_ptr).fetch_add(1, Ordering::Relaxed);

                let r_ptr = std::ptr::addr_of_mut!((*self.raw).read_head) as *mut AtomicU64;
                (*r_ptr).store(new_r, Ordering::Release);
            }
            r = new_r;
        }

        if r < w {
            let slot = (r as usize) % (SHM_RING_CAPACITY as usize);
            unsafe {
                let packet_ptr = &mut (*self.raw).packets[slot];
                Some(packet_ptr)
            }
        } else {
            None
        }
    }
}

unsafe impl Send for ShmRingBufferReader {}
unsafe impl Sync for ShmRingBufferReader {}
