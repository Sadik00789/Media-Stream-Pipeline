pub mod ffi;
pub mod memory;
pub mod pyo3_bindings;
pub mod ring_buffer;

use ffi::{
    AudioFramePacket, DetectionBox, DspConfig, DspWindowType, dsp_configure, dsp_init,
    dsp_process_frame, simd_fast_box_nms, simd_rgb24_to_planar_f32,
};
use std::sync::Mutex;

static DSP_INITIALIZED: Mutex<bool> = Mutex::new(false);

/// Safe RAII wrapper around C11 Hardware DSP Engine
pub struct DspEngine;

impl DspEngine {
    pub fn new() -> Result<Self, &'static str> {
        let mut init_guard = DSP_INITIALIZED
            .lock()
            .map_err(|_| "DSP initialization mutex poisoned")?;
        if !*init_guard {
            let res = unsafe { dsp_init() };
            if res != 0 {
                return Err("Failed to initialize C DSP subsystem");
            }
            *init_guard = true;
        }
        Ok(Self)
    }

    /// Configure runtime DSP parameters (VAD threshold, window function, sample rate).
    pub fn configure(&self, vad_threshold: f32, window_type: DspWindowType, sample_rate: u32) {
        let cfg = DspConfig {
            vad_threshold,
            window_type,
            sample_rate,
        };
        unsafe {
            dsp_configure(&cfg as *const DspConfig);
        }
    }

    /// Process single audio packet in-place: SIMD windowing + FFT + VAD
    pub fn process_frame(&self, packet: &mut AudioFramePacket) {
        unsafe {
            dsp_process_frame(packet as *mut AudioFramePacket);
        }
    }
}

/// Safe wrapper around C11 SIMD Vision Kernels
pub struct VisionSimd;

impl VisionSimd {
    pub fn rgb24_to_planar_f32(rgb_in: &[u8], planar_out: &mut [f32], width: usize, height: usize) {
        assert!(rgb_in.len() >= width * height * 3);
        assert!(planar_out.len() >= width * height * 3);
        unsafe {
            simd_rgb24_to_planar_f32(
                rgb_in.as_ptr(),
                planar_out.as_mut_ptr(),
                width as std::os::raw::c_int,
                height as std::os::raw::c_int,
            );
        }
    }

    pub fn fast_box_nms(
        boxes_in: &[DetectionBox],
        iou_threshold: f32,
        max_out: usize,
    ) -> Vec<DetectionBox> {
        let mut out = vec![DetectionBox::default(); max_out];
        let count = unsafe {
            simd_fast_box_nms(
                boxes_in.as_ptr(),
                boxes_in.len() as u32,
                iou_threshold,
                out.as_mut_ptr(),
                max_out as u32,
            )
        };
        out.truncate(count as usize);
        out
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
            sample_count: FRAME_SAMPLES,
            ..Default::default()
        };

        let mut pcm = [0.0f32; FRAME_SAMPLES as usize];
        for (i, sample) in pcm.iter_mut().enumerate() {
            *sample = (i as f32 * 0.01).sin();
        }
        packet.pcm_data = pcm;

        engine.process_frame(&mut packet);
        let first_bin = { packet.spectrogram[0] };
        assert_ne!(first_bin, 0.0);
    }

    #[test]
    fn test_vision_simd_rgb_and_nms() {
        let rgb = vec![128u8; 640 * 480 * 3];
        let mut planar = vec![0.0f32; 640 * 480 * 3];
        VisionSimd::rgb24_to_planar_f32(&rgb, &mut planar, 640, 480);
        assert!((planar[0] - (128.0 / 255.0)).abs() < 1e-4);

        let boxes = vec![
            DetectionBox {
                ymin: 0.1,
                xmin: 0.1,
                ymax: 0.5,
                xmax: 0.5,
                confidence: 0.9,
                class_id: 0,
            },
            DetectionBox {
                ymin: 0.11,
                xmin: 0.11,
                ymax: 0.49,
                xmax: 0.51,
                confidence: 0.8,
                class_id: 0,
            },
        ];
        let filtered = VisionSimd::fast_box_nms(&boxes, 0.45, 10);
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn test_abi_struct_sizes_and_dsp_configure() {
        assert_eq!(std::mem::size_of::<DetectionBox>(), 24);
        assert_eq!(std::mem::size_of::<AudioFramePacket>(), 3248);
        assert_eq!(std::mem::size_of::<ffi::VideoFramePacket>(), 922024);
        assert_eq!(
            std::mem::size_of::<ffi::SharedMemoryRingBuffer>(),
            48 + 64 * 3248
        );
        assert_eq!(
            std::mem::size_of::<ffi::SharedMemoryVideoRingBuffer>(),
            48 + 8 * 922024
        );

        let engine = DspEngine::new().expect("DSP init");
        engine.configure(0.02, DspWindowType::DSP_WINDOW_BLACKMAN, 16000);
        let mut packet = AudioFramePacket {
            sequence_number: 2,
            sample_rate: 16000,
            channels: 1,
            sample_count: FRAME_SAMPLES,
            ..Default::default()
        };
        let mut pcm = [0.0f32; FRAME_SAMPLES as usize];
        for (i, sample) in pcm.iter_mut().enumerate() {
            *sample = 0.5 * ((i as f32) * 2.0 * std::f32::consts::PI * 1000.0 / 16000.0).sin();
        }
        packet.pcm_data = pcm;
        engine.process_frame(&mut packet);
        let flags = { packet.flags };
        assert_ne!(flags & ffi::FLAG_DSP_DONE, 0);
        assert_eq!({ packet.vad_active }, 1);
        engine.configure(0.015, DspWindowType::DSP_WINDOW_HAMMING, 16000);
    }

    #[test]
    fn test_concurrent_dsp_engine_init() {
        let handles: Vec<_> = (0..4)
            .map(|_| std::thread::spawn(|| DspEngine::new().is_ok()))
            .collect();
        for h in handles {
            assert!(h.join().unwrap());
        }
    }
}
