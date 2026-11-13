use crate::ffi::{
    AudioFramePacket, SHM_MAGIC, SHM_RING_CAPACITY, SharedMemoryRingBuffer,
    SharedMemoryVideoRingBuffer, VIDEO_RING_CAPACITY, VIDEO_SHM_MAGIC, VideoFramePacket,
};
use std::sync::atomic::{AtomicU64, Ordering};

/// Safe zero-copy view over SharedMemoryRingBuffer (Audio)
pub struct ShmRingBufferReader {
    raw: *mut SharedMemoryRingBuffer,
}

impl ShmRingBufferReader {
    /// Creates a reader wrapper over an initialized shared-memory audio ring buffer.
    ///
    /// # Safety
    ///
    /// `raw` must be null or point to a valid, aligned, mapped `SharedMemoryRingBuffer`
    /// that outlives this reader.
    pub unsafe fn from_raw(raw: *mut SharedMemoryRingBuffer) -> Result<Self, &'static str> {
        if raw.is_null() {
            return Err("Shared memory pointer is null");
        }
        let magic = unsafe { (*raw).magic };
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

    /// Obtains a direct mutable pointer to a packet in shared memory without copying
    #[inline]
    pub fn get_packet_mut(&mut self, slot: usize) -> &mut AudioFramePacket {
        unsafe { &mut (*self.raw).packets[slot % (SHM_RING_CAPACITY as usize)] }
    }

    /// Read next available packet with automatic overrun recovery
    pub fn poll_next_packet(&mut self) -> Option<&mut AudioFramePacket> {
        let mut r = self.read_head();
        let w = self.write_head();

        if w > r + (SHM_RING_CAPACITY as u64) {
            // Overrun detected: writer wrapped around before reader caught up
            unsafe {
                let r_ptr = std::ptr::addr_of_mut!((*self.raw).read_head) as *mut AtomicU64;
                loop {
                    let new_r = w - (SHM_RING_CAPACITY as u64);
                    match (*r_ptr).compare_exchange_weak(
                        r,
                        new_r,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    ) {
                        Ok(_) => {
                            let dropped = new_r - r;
                            let drop_ptr = std::ptr::addr_of_mut!((*self.raw).dropped_frames)
                                as *mut AtomicU64;
                            (*drop_ptr).fetch_add(dropped, Ordering::Relaxed);

                            let overrun_ptr =
                                std::ptr::addr_of_mut!((*self.raw).overrun_count) as *mut AtomicU64;
                            (*overrun_ptr).fetch_add(1, Ordering::Relaxed);

                            r = new_r;
                            break;
                        }
                        Err(actual) => {
                            r = actual;
                            if w <= r + (SHM_RING_CAPACITY as u64) {
                                break;
                            }
                        }
                    }
                }
            }
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

/// Safe zero-copy view over SharedMemoryVideoRingBuffer (Video)
pub struct ShmVideoRingBufferReader {
    raw: *mut SharedMemoryVideoRingBuffer,
}

impl ShmVideoRingBufferReader {
    /// Creates a reader wrapper over an initialized shared-memory video ring buffer.
    ///
    /// # Safety
    ///
    /// `raw` must be null or point to a valid, aligned, mapped `SharedMemoryVideoRingBuffer`
    /// that outlives this reader.
    pub unsafe fn from_raw(raw: *mut SharedMemoryVideoRingBuffer) -> Result<Self, &'static str> {
        if raw.is_null() {
            return Err("Video shared memory pointer is null");
        }
        let magic = unsafe { (*raw).magic };
        if magic != VIDEO_SHM_MAGIC {
            return Err("Invalid video shared memory magic number");
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

    /// Obtains a direct mutable pointer to a video packet in shared memory without copying
    #[inline]
    pub fn get_packet_mut(&mut self, slot: usize) -> &mut VideoFramePacket {
        unsafe { &mut (*self.raw).packets[slot % (VIDEO_RING_CAPACITY as usize)] }
    }

    /// Read next available video frame packet with automatic overrun recovery
    pub fn poll_next_packet(&mut self) -> Option<&mut VideoFramePacket> {
        let mut r = self.read_head();
        let w = self.write_head();

        if w > r + (VIDEO_RING_CAPACITY as u64) {
            // Overrun detected: writer wrapped around before reader caught up
            unsafe {
                let r_ptr = std::ptr::addr_of_mut!((*self.raw).read_head) as *mut AtomicU64;
                loop {
                    let new_r = w - (VIDEO_RING_CAPACITY as u64);
                    match (*r_ptr).compare_exchange_weak(
                        r,
                        new_r,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    ) {
                        Ok(_) => {
                            let dropped = new_r - r;
                            let drop_ptr = std::ptr::addr_of_mut!((*self.raw).dropped_frames)
                                as *mut AtomicU64;
                            (*drop_ptr).fetch_add(dropped, Ordering::Relaxed);

                            let overrun_ptr =
                                std::ptr::addr_of_mut!((*self.raw).overrun_count) as *mut AtomicU64;
                            (*overrun_ptr).fetch_add(1, Ordering::Relaxed);

                            r = new_r;
                            break;
                        }
                        Err(actual) => {
                            r = actual;
                            if w <= r + (VIDEO_RING_CAPACITY as u64) {
                                break;
                            }
                        }
                    }
                }
            }
        }

        if r < w {
            let slot = (r as usize) % (VIDEO_RING_CAPACITY as usize);
            unsafe {
                let packet_ptr = &mut (*self.raw).packets[slot];
                Some(packet_ptr)
            }
        } else {
            None
        }
    }
}

unsafe impl Send for ShmVideoRingBufferReader {}
unsafe impl Sync for ShmVideoRingBufferReader {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::{SHM_VERSION, VIDEO_SHM_VERSION};

    fn alloc_zeroed_box<T>() -> Box<T> {
        let layout = std::alloc::Layout::new::<T>();
        unsafe {
            let ptr = std::alloc::alloc_zeroed(layout) as *mut T;
            assert!(!ptr.is_null(), "Allocation failed");
            Box::from_raw(ptr)
        }
    }

    #[test]
    fn test_audio_ring_buffer_validation_poll_and_overrun() {
        assert!(unsafe { ShmRingBufferReader::from_raw(std::ptr::null_mut()) }.is_err());

        let mut ring: Box<SharedMemoryRingBuffer> = alloc_zeroed_box();
        assert!(unsafe { ShmRingBufferReader::from_raw(&mut *ring) }.is_err());

        ring.magic = SHM_MAGIC;
        ring.version = SHM_VERSION;
        ring.buffer_capacity = SHM_RING_CAPACITY;
        ring.packet_size = std::mem::size_of::<AudioFramePacket>() as u32;

        let mut reader = unsafe { ShmRingBufferReader::from_raw(&mut *ring) }.unwrap();
        assert!(reader.poll_next_packet().is_none());

        // Write 3 frames
        for seq in 1..=3u64 {
            let slot = ((seq - 1) as usize) % (SHM_RING_CAPACITY as usize);
            ring.packets[slot].sequence_number = seq;
            ring.write_head = seq;
        }

        for expected in 1..=3u64 {
            let pkt = reader.poll_next_packet().expect("expected packet");
            assert_eq!({ pkt.sequence_number }, expected);
            reader.advance_read_head();
        }
        assert!(reader.poll_next_packet().is_none());

        // Trigger overrun: advance write_head beyond capacity
        let overrun_w = 3 + SHM_RING_CAPACITY as u64 + 5;
        for seq in 4..=overrun_w {
            let slot = ((seq - 1) as usize) % (SHM_RING_CAPACITY as usize);
            ring.packets[slot].sequence_number = seq;
        }
        ring.write_head = overrun_w;

        let pkt = reader
            .poll_next_packet()
            .expect("expected overrun-recovered packet");
        assert_eq!(
            { pkt.sequence_number },
            overrun_w - SHM_RING_CAPACITY as u64 + 1
        );
        assert_eq!({ ring.overrun_count }, 1);
        assert_eq!({ ring.dropped_frames }, 5);
    }

    #[test]
    fn test_video_ring_buffer_validation_poll_and_overrun() {
        assert!(unsafe { ShmVideoRingBufferReader::from_raw(std::ptr::null_mut()) }.is_err());

        let mut ring: Box<SharedMemoryVideoRingBuffer> = alloc_zeroed_box();
        assert!(unsafe { ShmVideoRingBufferReader::from_raw(&mut *ring) }.is_err());

        ring.magic = VIDEO_SHM_MAGIC;
        ring.version = VIDEO_SHM_VERSION;
        ring.buffer_capacity = VIDEO_RING_CAPACITY;
        ring.packet_size = std::mem::size_of::<VideoFramePacket>() as u32;

        let mut reader = unsafe { ShmVideoRingBufferReader::from_raw(&mut *ring) }.unwrap();
        assert!(reader.poll_next_packet().is_none());

        ring.packets[0].sequence_number = 10;
        ring.write_head = 1;
        let pkt = reader.poll_next_packet().unwrap();
        assert_eq!({ pkt.sequence_number }, 10);
        reader.advance_read_head();
        assert!(reader.poll_next_packet().is_none());

        // Trigger overrun
        let overrun_w = 1 + VIDEO_RING_CAPACITY as u64 + 3;
        for i in 1..overrun_w {
            let slot = (i as usize) % (VIDEO_RING_CAPACITY as usize);
            ring.packets[slot].sequence_number = i + 1;
        }
        ring.write_head = overrun_w;
        let recovered = reader.poll_next_packet().unwrap();
        assert_eq!(
            { recovered.sequence_number },
            overrun_w - VIDEO_RING_CAPACITY as u64 + 1
        );
        assert_eq!({ ring.overrun_count }, 1);
        assert_eq!({ ring.dropped_frames }, 3);
    }
}
