use crate::py_runtime::{PyRuntimeManager, VisionRuntimeManager};
use tensor_bridge::ffi::{AudioFramePacket, DetectionBox, VideoFramePacket, FLAG_ML_DONE, MAX_DETECTIONS};
use tensor_bridge::{DspEngine, VisionSimd};

pub struct AudioPipeline {
    dsp: DspEngine,
    py_runtime: Option<PyRuntimeManager>,
}

impl AudioPipeline {
    pub fn new(model_path: &str) -> Result<Self, String> {
        let dsp = DspEngine::new().map_err(|e| format!("DSP initialization error: {}", e))?;
        
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

        // 2. Embedded Python ML Runtime (zero-copy view)
        if let Some(ref py_rt) = self.py_runtime {
            let pcm = packet.pcm_data;
            let (confidence, transcript) = py_rt.process_frame(&pcm);
            packet.ml_confidence = confidence;

            // Copy transcript string into fixed-size buffer safely
            let bytes = transcript.as_bytes();
            let copy_len = bytes.len().min(packet.ml_transcript.len() - 1);
            for i in 0..copy_len {
                packet.ml_transcript[i] = bytes[i] as std::os::raw::c_char;
            }
            packet.ml_transcript[copy_len] = 0; // null-terminate

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
            let pixel_data = packet.pixel_data;
            let detections = vision_rt.process_frame(&pixel_data);

            // Apply fast SIMD NMS filtering
            let filtered = VisionSimd::fast_box_nms(&detections, 0.45, MAX_DETECTIONS as usize);
            let count = filtered.len().min(MAX_DETECTIONS as usize);
            packet.detection_count = count as u32;

            for i in 0..count {
                packet.detections[i] = filtered[i];
            }
            for i in count..(MAX_DETECTIONS as usize) {
                packet.detections[i] = DetectionBox::default();
            }

            packet.flags |= FLAG_ML_DONE;
        }
    }
}
