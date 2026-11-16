use std::ffi::c_void;
use std::fs::{File, OpenOptions};
use std::path::PathBuf;
use tensor_bridge::ffi::{
    AudioFramePacket, SHM_MAGIC, SHM_RING_CAPACITY, SHM_VERSION, SharedMemoryRingBuffer,
    SharedMemoryVideoRingBuffer, VIDEO_RING_CAPACITY, VIDEO_SHM_MAGIC, VIDEO_SHM_VERSION,
    VideoFramePacket,
};
use tensor_bridge::ring_buffer::{ShmRingBufferReader, ShmVideoRingBufferReader};
use tracing::info;

#[cfg(unix)]
mod platform {
    use super::*;
    use nix::sys::mman::{MapFlags, ProtFlags, mmap, munmap};
    use std::num::NonZeroUsize;
    use std::os::unix::fs::OpenOptionsExt;
    use std::ptr::NonNull;

    pub struct MmapRegion {
        pub ptr: *mut c_void,
        pub size: usize,
        _file: File,
    }

    pub fn shm_path(shm_name: &str) -> PathBuf {
        let clean_name = shm_name.trim_start_matches('/');
        PathBuf::from("/dev/shm").join(clean_name)
    }

    pub fn map_file(shm_path: &PathBuf, size: usize) -> Result<MmapRegion, String> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o666)
            .open(shm_path)
            .map_err(|e| {
                format!(
                    "Failed to open shared memory file '{}': {}",
                    shm_path.display(),
                    e
                )
            })?;

        let metadata = file
            .metadata()
            .map_err(|e| format!("Failed to get shm metadata: {}", e))?;

        if metadata.len() < size as u64 {
            file.set_len(size as u64)
                .map_err(|e| format!("Failed to set shm size to {} bytes: {}", size, e))?;
            info!(
                "Initialized shared memory file '{}' with size {} bytes",
                shm_path.display(),
                size
            );
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

        Ok(MmapRegion {
            ptr: mmap_ptr.as_ptr(),
            size,
            _file: file,
        })
    }

    impl Drop for MmapRegion {
        fn drop(&mut self) {
            if let Some(non_null) = NonNull::new(self.ptr) {
                unsafe {
                    let _ = munmap(non_null, self.size);
                }
            }
        }
    }
}

#[cfg(windows)]
mod platform {
    use super::*;
    use std::os::windows::io::AsRawHandle;
    use std::ptr;

    const PAGE_READWRITE: u32 = 0x04;
    const FILE_MAP_ALL_ACCESS: u32 = 0x000F001F;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn CreateFileMappingW(
            h_file: *mut c_void,
            lp_file_mapping_attributes: *const c_void,
            fl_protect: u32,
            dw_maximum_size_high: u32,
            dw_maximum_size_low: u32,
            lp_name: *const u16,
        ) -> *mut c_void;
        fn MapViewOfFile(
            h_file_mapping_object: *mut c_void,
            dw_desired_access: u32,
            dw_file_offset_high: u32,
            dw_file_offset_low: u32,
            dw_number_of_bytes_to_map: usize,
        ) -> *mut c_void;
        fn UnmapViewOfFile(lp_base_address: *const c_void) -> i32;
        fn CloseHandle(h_object: *mut c_void) -> i32;
    }

    pub struct MmapRegion {
        pub ptr: *mut c_void,
        mapping_handle: *mut c_void,
        _file: File,
    }

    pub fn shm_path(shm_name: &str) -> PathBuf {
        let clean_name = shm_name.trim_start_matches('/');
        std::env::temp_dir().join(clean_name)
    }

    pub fn map_file(shm_path: &PathBuf, size: usize) -> Result<MmapRegion, String> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(shm_path)
            .map_err(|e| {
                format!(
                    "Failed to open shared memory file '{}': {}",
                    shm_path.display(),
                    e
                )
            })?;

        let metadata = file
            .metadata()
            .map_err(|e| format!("Failed to get shm metadata: {}", e))?;

        if metadata.len() < size as u64 {
            file.set_len(size as u64)
                .map_err(|e| format!("Failed to set shm size to {} bytes: {}", size, e))?;
            info!(
                "Initialized shared memory file '{}' with size {} bytes",
                shm_path.display(),
                size
            );
        }

        let high = ((size as u64) >> 32) as u32;
        let low = (size as u64) as u32;
        let mapping_handle = unsafe {
            CreateFileMappingW(
                file.as_raw_handle(),
                ptr::null(),
                PAGE_READWRITE,
                high,
                low,
                ptr::null(),
            )
        };
        if mapping_handle.is_null() {
            return Err(format!(
                "CreateFileMappingW failed for '{}': {}",
                shm_path.display(),
                std::io::Error::last_os_error()
            ));
        }

        let view_ptr = unsafe { MapViewOfFile(mapping_handle, FILE_MAP_ALL_ACCESS, 0, 0, size) };
        if view_ptr.is_null() {
            unsafe {
                CloseHandle(mapping_handle);
            }
            return Err(format!(
                "MapViewOfFile failed for '{}': {}",
                shm_path.display(),
                std::io::Error::last_os_error()
            ));
        }

        Ok(MmapRegion {
            ptr: view_ptr,
            mapping_handle,
            _file: file,
        })
    }

    impl Drop for MmapRegion {
        fn drop(&mut self) {
            unsafe {
                if !self.ptr.is_null() {
                    let _ = UnmapViewOfFile(self.ptr);
                }
                if !self.mapping_handle.is_null() {
                    let _ = CloseHandle(self.mapping_handle);
                }
            }
        }
    }
}

pub struct ShmContext {
    _region: platform::MmapRegion,
}

impl ShmContext {
    pub fn attach(shm_name: &str) -> Result<(Self, ShmRingBufferReader), String> {
        let shm_path = platform::shm_path(shm_name);
        let size = std::mem::size_of::<SharedMemoryRingBuffer>();
        let region = platform::map_file(&shm_path, size)?;
        let ring_ptr = region.ptr as *mut SharedMemoryRingBuffer;

        // Initialize header if magic is uninitialized
        unsafe {
            if (*ring_ptr).magic != SHM_MAGIC {
                (*ring_ptr).magic = SHM_MAGIC;
                (*ring_ptr).version = SHM_VERSION;
                (*ring_ptr).buffer_capacity = SHM_RING_CAPACITY;
                (*ring_ptr).packet_size = std::mem::size_of::<AudioFramePacket>() as u32;
                (*ring_ptr).write_head = 0;
                (*ring_ptr).read_head = 0;
                (*ring_ptr).dropped_frames = 0;
                (*ring_ptr).overrun_count = 0;
            }
        }

        let reader = unsafe {
            ShmRingBufferReader::from_raw(ring_ptr)
                .map_err(|e| format!("Failed to initialize SHM reader: {}", e))?
        };

        Ok((Self { _region: region }, reader))
    }
}

pub struct VideoShmContext {
    _region: platform::MmapRegion,
}

impl VideoShmContext {
    pub fn attach(shm_name: &str) -> Result<(Self, ShmVideoRingBufferReader), String> {
        let shm_path = platform::shm_path(shm_name);
        let size = std::mem::size_of::<SharedMemoryVideoRingBuffer>();
        let region = platform::map_file(&shm_path, size)?;
        let ring_ptr = region.ptr as *mut SharedMemoryVideoRingBuffer;

        // Initialize header if magic is uninitialized
        unsafe {
            if (*ring_ptr).magic != VIDEO_SHM_MAGIC {
                (*ring_ptr).magic = VIDEO_SHM_MAGIC;
                (*ring_ptr).version = VIDEO_SHM_VERSION;
                (*ring_ptr).buffer_capacity = VIDEO_RING_CAPACITY;
                (*ring_ptr).packet_size = std::mem::size_of::<VideoFramePacket>() as u32;
                (*ring_ptr).write_head = 0;
                (*ring_ptr).read_head = 0;
                (*ring_ptr).dropped_frames = 0;
                (*ring_ptr).overrun_count = 0;
            }
        }

        let reader = unsafe {
            ShmVideoRingBufferReader::from_raw(ring_ptr)
                .map_err(|e| format!("Failed to initialize Video SHM reader: {}", e))?
        };

        Ok((Self { _region: region }, reader))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_and_video_shm_context_attach() {
        let pid = std::process::id();
        let audio_name = format!("/test_audio_shm_{}", pid);
        let video_name = format!("/test_video_shm_{}", pid);

        let (_audio_ctx, mut audio_reader) =
            ShmContext::attach(&audio_name).expect("Audio SHM attach failed");
        assert_eq!(audio_reader.write_head(), 0);
        assert_eq!(audio_reader.read_head(), 0);
        assert!(audio_reader.poll_next_packet().is_none());

        let (_video_ctx, mut video_reader) =
            VideoShmContext::attach(&video_name).expect("Video SHM attach failed");
        assert_eq!(video_reader.write_head(), 0);
        assert_eq!(video_reader.read_head(), 0);
        assert!(video_reader.poll_next_packet().is_none());

        let _ = std::fs::remove_file(platform::shm_path(&audio_name));
        let _ = std::fs::remove_file(platform::shm_path(&video_name));
    }
}
