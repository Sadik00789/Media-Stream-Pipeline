use nix::sys::mman::{mmap, munmap, MapFlags, ProtFlags};
use std::ffi::c_void;
use std::fs::OpenOptions;
use std::num::NonZeroUsize;
use std::os::unix::fs::OpenOptionsExt;
use std::path::PathBuf;
use std::ptr::NonNull;
use tensor_bridge::ffi::{SharedMemoryRingBuffer, SHM_MAGIC, SHM_RING_CAPACITY, SHM_VERSION};
use tensor_bridge::ring_buffer::ShmRingBufferReader;
use tracing::info;

pub struct ShmContext {
    ptr: *mut SharedMemoryRingBuffer,
    size: usize,
}

impl ShmContext {
    pub fn attach(shm_name: &str) -> Result<(Self, ShmRingBufferReader), String> {
        let clean_name = shm_name.trim_start_matches('/');
        let shm_path = PathBuf::from("/dev/shm").join(clean_name);

        let size = std::mem::size_of::<SharedMemoryRingBuffer>();

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .mode(0o666)
            .open(&shm_path)
            .map_err(|e| format!("Failed to open shared memory file '{}': {}", shm_path.display(), e))?;

        let metadata = file
            .metadata()
            .map_err(|e| format!("Failed to get shm metadata: {}", e))?;

        if metadata.len() < size as u64 {
            file.set_len(size as u64)
                .map_err(|e| format!("Failed to set shm size to {} bytes: {}", size, e))?;
            info!("Initialized shared memory file '{}' with size {} bytes", shm_path.display(), size);
        }

        let non_zero_size = NonZeroUsize::new(size).ok_or("Invalid SHM struct size")?;

        let mmap_ptr = unsafe {
            mmap(
                None,
                non_zero_size,
                ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
                MapFlags::MAP_SHARED,
                &file,
                0,
            )
            .map_err(|e| format!("Failed to mmap shared memory segment: {}", e))?
        };

        let ring_ptr = mmap_ptr.as_ptr() as *mut SharedMemoryRingBuffer;

        // Initialize header if magic is uninitialized
        unsafe {
            if (*ring_ptr).magic != SHM_MAGIC {
                (*ring_ptr).magic = SHM_MAGIC;
                (*ring_ptr).version = SHM_VERSION;
                (*ring_ptr).buffer_capacity = SHM_RING_CAPACITY as u32;
                (*ring_ptr).packet_size = std::mem::size_of::<tensor_bridge::ffi::AudioFramePacket>() as u32;
                (*ring_ptr).write_head = 0;
                (*ring_ptr).read_head = 0;
            }
        }

        let reader = unsafe {
            ShmRingBufferReader::from_raw(ring_ptr)
                .map_err(|e| format!("Failed to initialize SHM reader: {}", e))?
        };

        Ok((
            Self {
                ptr: ring_ptr,
                size,
            },
            reader,
        ))
    }
}

impl Drop for ShmContext {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            if let Some(non_null) = NonNull::new(self.ptr as *mut c_void) {
                unsafe {
                    let _ = munmap(non_null, self.size);
                }
            }
        }
    }
}
