#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

// Ensure AudioFramePacket and SharedMemoryRingBuffer traits
unsafe impl Send for AudioFramePacket {}
unsafe impl Sync for AudioFramePacket {}

unsafe impl Send for SharedMemoryRingBuffer {}
unsafe impl Sync for SharedMemoryRingBuffer {}
