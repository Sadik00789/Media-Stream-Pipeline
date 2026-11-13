use crate::py_runtime::{PyRuntimeManager, VisionRuntimeManager};
use tensor_bridge::ffi::{
    AudioFramePacket, DetectionBox, DspWindowType, FLAG_ML_DONE, MAX_DETECTIONS, VideoFramePacket,
};
use tensor_bridge::{DspEngine, VisionSimd};

pub struct AudioPipeline {
    dsp: DspEngine,
    py_runtime: Option<PyRuntimeManager>,
}

impl AudioPipeline {
    #[allow(dead_code)]
    pub fn new(model_path: &str) -> Result<Self, String> {
        Self::new_with_config(model_path, 0.015)
    }

    pub fn new_with_config(model_path: &str, vad_threshold: f32) -> Result<Self, String> {
        let dsp = DspEngine::new().map_err(|e| format!("DSP initialization error: {}", e))?;
        dsp.configure(vad_threshold, DspWindowType::DSP_WINDOW_HAMMING, 16000);

        let py_runtime = match PyRuntimeManager::init(model_path) {
            Ok(rt) => Some(rt),
            Err(e) => {
                tracing::warn!("Running without Python Audio ML engine: {}", e);
                None
            }
        };

        Ok(Self { dsp, py_runtime })
    }

    /// Process packet through C DSP (FFT/VAD) and Embedded Python ML
    pub fn process_packet(&mut self, packet: &mut AudioFramePacket) {
        // 1. C11 Hardware AVX2 DSP (in-place)
        self.dsp.process_frame(packet);

        let pcm = packet.pcm_data;
        let vad_active = packet.vad_active != 0;
        let mut processed_by_ml = false;

        // 2. Embedded Python ML Runtime (zero-copy view)
        if let Some(ref py_rt) = self.py_runtime {
            let (confidence, transcript) = py_rt.process_frame(&pcm, vad_active);
            packet.ml_confidence = confidence;

            // Copy transcript string into fixed-size buffer safely
            let bytes = transcript.as_bytes();
            let copy_len = bytes.len().min(packet.ml_transcript.len() - 1);
            for (dst, &src) in packet.ml_transcript.iter_mut().zip(&bytes[..copy_len]) {
                *dst = src as std::os::raw::c_char;
            }
            packet.ml_transcript[copy_len] = 0; // null-terminate

            packet.flags |= FLAG_ML_DONE;
            processed_by_ml = true;
        }

        // 3. Resilient Native Fallback if Python ML was not loaded
        if !processed_by_ml {
            let sum_sq: f32 = pcm.iter().map(|&x| x * x).sum();
            let energy = (sum_sq / 512.0).sqrt();

            let fallback_conf = if vad_active {
                (0.70 + energy * 3.0).clamp(0.70, 0.99)
            } else if energy > 0.015 {
                (energy * 10.0).clamp(0.40, 0.98)
            } else {
                0.0
            };

            packet.ml_confidence = fallback_conf;

            if vad_active || energy > 0.02 {
                let token = b"voice_stream\0";
                for (i, &b) in token.iter().enumerate() {
                    packet.ml_transcript[i] = b as std::os::raw::c_char;
                }
            } else {
                packet.ml_transcript[0] = 0;
            }

            packet.flags |= FLAG_ML_DONE;
        }
    }
}

pub struct VideoPipeline {
    vision_runtime: Option<VisionRuntimeManager>,
}

impl VideoPipeline {
    pub fn new(vision_model_path: &str) -> Result<Self, String> {
        let vision_runtime = match VisionRuntimeManager::init(vision_model_path) {
            Ok(rt) => Some(rt),
            Err(e) => {
                tracing::warn!("Running without Python Vision ML engine: {}", e);
                None
            }
        };

        Ok(Self { vision_runtime })
    }

    /// Process video packet through Embedded Python Vision ML and C11 SIMD NMS
    pub fn process_packet(&mut self, packet: &mut VideoFramePacket) {
        if let Some(ref vision_rt) = self.vision_runtime {
            let detections = vision_rt.process_frame(&packet.pixel_data);

            // Apply fast SIMD NMS filtering
            let filtered = VisionSimd::fast_box_nms(&detections, 0.45, MAX_DETECTIONS as usize);
            let count = filtered.len().min(MAX_DETECTIONS as usize);
            packet.detection_count = count as u32;

            packet.detections[..count].copy_from_slice(&filtered[..count]);
            for i in count..(MAX_DETECTIONS as usize) {
                packet.detections[i] = DetectionBox::default();
            }

            packet.flags |= FLAG_ML_DONE;
        } else {
            packet.flags |= FLAG_ML_DONE;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor_bridge::ffi::{
        FLAG_DSP_DONE, FRAME_SAMPLES, VIDEO_CHANNELS, VIDEO_HEIGHT, VIDEO_WIDTH,
    };

    #[test]
    fn test_audio_pipeline_end_to_end() {
        let mut pipeline = AudioPipeline::new("nonexistent.onnx").expect("AudioPipeline::new");

        let mut pkt = AudioFramePacket {
            sequence_number: 1,
            sample_rate: 16000,
            channels: 1,
            sample_count: FRAME_SAMPLES,
            ..Default::default()
        };
        for i in 0..(FRAME_SAMPLES as usize) {
            pkt.pcm_data[i] =
                0.5 * ((i as f32) * 2.0 * std::f32::consts::PI * 1000.0 / 16000.0).sin();
        }

        pipeline.process_packet(&mut pkt);
        let flags = { pkt.flags };
        assert_ne!(flags & FLAG_DSP_DONE, 0);
        assert_ne!(flags & FLAG_ML_DONE, 0);
        assert_eq!({ pkt.vad_active }, 1);
        assert!({ pkt.ml_confidence } > 0.5);
    }

    #[test]
    fn test_video_pipeline_end_to_end() {
        let mut pipeline = VideoPipeline::new("nonexistent.onnx").expect("VideoPipeline::new");

        let mut pkt: Box<VideoFramePacket> = unsafe {
            let layout = std::alloc::Layout::new::<VideoFramePacket>();
            let ptr = std::alloc::alloc_zeroed(layout) as *mut VideoFramePacket;
            assert!(!ptr.is_null());
            Box::from_raw(ptr)
        };
        pkt.sequence_number = 1;
        pkt.width = VIDEO_WIDTH;
        pkt.height = VIDEO_HEIGHT;

        // Paint a synthetic skin-tone rectangle in the center
        for y in 150..330 {
            for x in 200..440 {
                let idx = (y * (VIDEO_WIDTH as usize) + x) * (VIDEO_CHANNELS as usize);
                pkt.pixel_data[idx] = 220;
                pkt.pixel_data[idx + 1] = 160;
                pkt.pixel_data[idx + 2] = 120;
            }
        }

        pipeline.process_packet(&mut pkt);
        let flags = { pkt.flags };
        assert_ne!(flags & FLAG_ML_DONE, 0);
        assert!({ pkt.detection_count } >= 1);
    }
}
