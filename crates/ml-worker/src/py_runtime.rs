use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySequence, PyTuple};
use std::path::PathBuf;
use tensor_bridge::ffi::DetectionBox;
use tensor_bridge::pyo3_bindings::{as_pyarray_video, slice_to_pylist};
use tracing::{error, info, warn};

pub struct PyRuntimeManager {
    engine_obj: Py<PyAny>,
}

impl PyRuntimeManager {
    pub fn init(_model_path: &str) -> Result<Self, String> {
        info!("Initializing embedded Python Audio ML runtime (PyO3)...");

        Python::with_gil(|py| {
            let sys = py.import_bound("sys").map_err(|e| e.to_string())?;
            let path: Bound<'_, PyList> = sys.getattr("path")
                .map_err(|e| e.to_string())?
                .extract()
                .map_err(|e| e.to_string())?;

            let mut candidates = vec![
                PathBuf::from("python/src"),
                PathBuf::from("../../python/src"),
                PathBuf::from("../python/src"),
                PathBuf::from("python/src/streaming_ml"),
                PathBuf::from("../../python/src/streaming_ml"),
            ];

            if let Ok(cwd) = std::env::current_dir() {
                candidates.push(cwd.join("python/src"));
                candidates.push(cwd.join("../../python/src"));
                candidates.push(cwd.join("../python/src"));
                candidates.push(cwd.join("python/src/streaming_ml"));
                candidates.push(cwd.join("../../python/src/streaming_ml"));
                if let Some(parent) = cwd.parent() {
                    candidates.push(parent.join("python/src"));
                    candidates.push(parent.join("python/src/streaming_ml"));
                    if let Some(p2) = parent.parent() {
                        candidates.push(p2.join("python/src"));
                        candidates.push(p2.join("python/src/streaming_ml"));
                    }
                }
            }

            for cand in candidates {
                if cand.exists() {
                    if let Ok(canon) = cand.canonicalize() {
                        let _ = path.insert(0, canon.to_string_lossy().to_string());
                    } else {
                        let _ = path.insert(0, cand.to_string_lossy().to_string());
                    }
                }
            }

            // Attempt importing as module or direct script with clear traceback logging
            let module = py.import_bound("streaming_ml.asr_engine")
                .or_else(|_| py.import_bound("asr_engine"))
                .or_else(|_| py.import_bound("streaming_ml.inference"))
                .or_else(|_| py.import_bound("inference"))
                .map_err(|e| {
                    error!("Python audio ML import failure: {:?}", e);
                    format!("Failed to import audio ML module: {:?}", e)
                })?;

            let instance = if let Ok(engine_class) = module.getattr("StreamingAsrEngine") {
                engine_class.call0()
                    .map_err(|e| format!("Failed to call StreamingAsrEngine(): {:?}", e))?
            } else if let Ok(engine_class) = module.getattr("StreamingInferenceEngine") {
                engine_class.call0()
                    .map_err(|e| format!("Failed to call StreamingInferenceEngine(): {:?}", e))?
            } else {
                return Err("Failed to find engine class in audio ML module".to_string());
            };

            info!("Embedded Python Audio ML Runtime successfully initialized");
            Ok(Self {
                engine_obj: instance.into(),
            })
        })
    }

    /// Executes ML inference on a raw slice of PCM audio with VAD context
    pub fn process_frame(&self, pcm_data: &[f32], vad_active: bool) -> (f32, String) {
        Python::with_gil(|py| {
            let py_list = slice_to_pylist(py, pcm_data);
            let engine = self.engine_obj.bind(py);

            match engine.call_method1("process_frame", (py_list, vad_active)) {
                Ok(result) => {
                    if let Ok(tuple) = result.extract::<Bound<'_, PyTuple>>() {
                        let conf: f32 = tuple.get_item(0).and_then(|v| v.extract()).unwrap_or(0.0);
                        let text: String = tuple.get_item(1).and_then(|v| v.extract()).unwrap_or_default();
                        (conf, text)
                    } else if let Ok(conf) = result.extract::<f32>() {
                        (conf, String::new())
                    } else {
                        (0.0, String::new())
                    }
                }
                Err(e) => {
                    warn!("Python audio inference error: {}", e);
                    (0.0, String::new())
                }
            }
        })
    }
}

pub struct VisionRuntimeManager {
    engine_obj: Py<PyAny>,
}

impl VisionRuntimeManager {
    pub fn init(model_path: &str) -> Result<Self, String> {
        info!("Initializing embedded Python Vision ML runtime (PyO3)...");

        Python::with_gil(|py| {
            let sys = py.import_bound("sys").map_err(|e| e.to_string())?;
            let path: Bound<'_, PyList> = sys.getattr("path")
                .map_err(|e| e.to_string())?
                .extract()
                .map_err(|e| e.to_string())?;
            
            let mut candidates = vec![
                PathBuf::from("python/src"),
                PathBuf::from("../../python/src"),
                PathBuf::from("../python/src"),
                PathBuf::from("python/src/streaming_ml"),
                PathBuf::from("../../python/src/streaming_ml"),
            ];
            if let Ok(cwd) = std::env::current_dir() {
                candidates.push(cwd.join("python/src"));
                candidates.push(cwd.join("../../python/src"));
                candidates.push(cwd.join("../python/src"));
                candidates.push(cwd.join("python/src/streaming_ml"));
                candidates.push(cwd.join("../../python/src/streaming_ml"));
                if let Some(parent) = cwd.parent() {
                    candidates.push(parent.join("python/src"));
                    candidates.push(parent.join("python/src/streaming_ml"));
                    if let Some(p2) = parent.parent() {
                        candidates.push(p2.join("python/src"));
                        candidates.push(p2.join("python/src/streaming_ml"));
                    }
                }
            }

            for cand in candidates {
                if cand.exists() {
                    if let Ok(canon) = cand.canonicalize() {
                        let _ = path.insert(0, canon.to_string_lossy().to_string());
                    } else {
                        let _ = path.insert(0, cand.to_string_lossy().to_string());
                    }
                }
            }

            let vision_module = py.import_bound("streaming_ml.vision_engine")
                .or_else(|_| py.import_bound("vision_engine"))
                .map_err(|e| {
                    error!("Python vision ML import failure: {:?}", e);
                    format!("Failed to import streaming_ml.vision_engine: {:?}", e)
                })?;

            let engine_class = vision_module.getattr("VisionEngine")
                .map_err(|e| format!("Failed to get VisionEngine class: {}", e))?;

            let kwargs = PyDict::new_bound(py);
            let _ = kwargs.set_item("model_path", model_path);

            let instance = engine_class.call((), Some(&kwargs))
                .map_err(|e| format!("Failed to instantiate VisionEngine: {}", e))?;

            info!("Embedded Python Vision ML Runtime successfully initialized");
            Ok(Self {
                engine_obj: instance.into(),
            })
        })
    }

    /// Executes Vision inference on a zero-copy RGB24 pixel frame
    pub fn process_frame(&self, pixel_data: &[u8]) -> Vec<DetectionBox> {
        Python::with_gil(|py| {
            let py_arr = match as_pyarray_video(py, pixel_data) {
                Ok(arr) => arr,
                Err(e) => {
                    warn!("Failed to create zero-copy PyArray3 for video frame: {}", e);
                    return Vec::new();
                }
            };

            let engine = self.engine_obj.bind(py);

            match engine.call_method1("process_frame", (py_arr,)) {
                Ok(result) => {
                    let mut boxes = Vec::new();
                    if let Ok(seq) = result.extract::<Bound<'_, PySequence>>() {
                        let len = seq.len().unwrap_or(0);
                        for i in 0..len {
                            if let Ok(item) = seq.get_item(i) {
                                if let Ok(box_vals) = item.extract::<Vec<f32>>() {
                                    if box_vals.len() >= 5 {
                                        let class_id = if box_vals.len() >= 6 {
                                            box_vals[5] as u32
                                        } else {
                                            0
                                        };
                                        boxes.push(DetectionBox {
                                            ymin: box_vals[0],
                                            xmin: box_vals[1],
                                            ymax: box_vals[2],
                                            xmax: box_vals[3],
                                            confidence: box_vals[4],
                                            class_id,
                                        });
                                    }
                                }
                            }
                        }
                    }
                    boxes
                }
                Err(e) => {
                    warn!("Python vision inference execution error: {}", e);
                    Vec::new()
                }
            }
        })
    }
}
