pub mod ffi;
pub mod memory;
pub mod pyo3_bindings;
pub mod ring_buffer;

use ffi::{dsp_init, dsp_process_frame, AudioFramePacket};
use std::sync::atomic::{AtomicBool, Ordering};

static DSP_INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Safe RAII wrapper around C11 Hardware DSP Engine
pub struct DspEngine;

impl DspEngine {
    pub fn new() -> Result<Self, &'static str> {
        if !DSP_INITIALIZED.swap(true, Ordering::SeqCst) {
            let res = unsafe { dsp_init() };
            if res != 0 {
                DSP_INITIALIZED.store(false, Ordering::SeqCst);
                return Err("Failed to initialize C DSP subsystem");
            }
        }
        Ok(Self)
    }

    /// Process single audio packet in-place: SIMD windowing + FFT + VAD
    pub fn process_frame(&self, packet: &mut AudioFramePacket) {
        unsafe {
            dsp_process_frame(packet as *mut AudioFramePacket);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ffi::FRAME_SAMPLES;

    #[test]
    fn test_dsp_engine_process_frame() {
        let engine = DspEngine::new().expect("Failed to create DSP engine");
        let mut packet = AudioFramePacket {
            sequence_number: 1,
            timestamp_ns: 1000,
            sample_rate: 16000,
            channels: 1,
            sample_count: FRAME_SAMPLES as u32,
            ..Default::default()
        };

        // Cast loop upper-bound to usize so `i` indexes [f32] correctly
        for i in 0..(FRAME_SAMPLES as usize) {
            packet.pcm_data[i] = (i as f32 * 0.01).sin();
        }

        engine.process_frame(&mut packet);
        let first_bin = packet.spectrogram[0];
        assert_ne!(first_bin, 0.0);
    }
}