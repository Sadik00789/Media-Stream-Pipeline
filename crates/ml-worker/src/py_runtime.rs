use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySequence, PyTuple};
use tensor_bridge::ffi::DetectionBox;
use tensor_bridge::pyo3_bindings::{as_pyarray_video, slice_to_pylist};
use tracing::{info, warn};

pub struct PyRuntimeManager {
    engine_obj: Py<PyAny>,
}

impl PyRuntimeManager {
    pub fn init(model_path: &str) -> Result<Self, String> {
        info!("Initializing embedded Python Audio ML runtime (PyO3)...");

        Python::with_gil(|py| {
            // Add python/src to sys.path
            let sys = py.import_bound("sys").map_err(|e| e.to_string())?;
            let path: Bound<'_, PyList> = sys.getattr("path")
                .map_err(|e| e.to_string())?
                .extract()
                .map_err(|e| e.to_string())?;
            
            path.insert(0, "python/src").map_err(|e| e.to_string())?;
            path.insert(0, "../../python/src").map_err(|e| e.to_string())?;

            // Import streaming_ml.inference module
            let ml_module = py.import_bound("streaming_ml.inference")
                .map_err(|e| format!("Failed to import streaming_ml.inference: {}", e))?;

            let engine_class = ml_module.getattr("StreamingInferenceEngine")
                .map_err(|e| format!("Failed to get StreamingInferenceEngine class: {}", e))?;

            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("model_path", model_path)
                .map_err(|e| e.to_string())?;

            let instance = engine_class.call((), Some(&kwargs))
                .map_err(|e| format!("Failed to instantiate StreamingInferenceEngine: {}", e))?;

            info!("Embedded Python Audio ML Runtime successfully initialized");
            Ok(Self {
                engine_obj: instance.into(),
            })
        })
    }

    /// Executes ML inference on a raw slice of PCM audio
    pub fn process_frame(&self, pcm_data: &[f32]) -> (f32, String) {
        Python::with_gil(|py| {
            let py_list = slice_to_pylist(py, pcm_data);
            let engine = self.engine_obj.bind(py);

            match engine.call_method1("process_frame", (py_list,)) {
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
                    warn!("Python audio inference execution error: {}", e);
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
            // Add python/src to sys.path
            let sys = py.import_bound("sys").map_err(|e| e.to_string())?;
            let path: Bound<'_, PyList> = sys.getattr("path")
                .map_err(|e| e.to_string())?
                .extract()
                .map_err(|e| e.to_string())?;
            
            path.insert(0, "python/src").map_err(|e| e.to_string())?;
            path.insert(0, "../../python/src").map_err(|e| e.to_string())?;

            // Import streaming_ml.vision_engine module
            let vision_module = py.import_bound("streaming_ml.vision_engine")
                .map_err(|e| format!("Failed to import streaming_ml.vision_engine: {}", e))?;

            let engine_class = vision_module.getattr("VisionEngine")
                .map_err(|e| format!("Failed to get VisionEngine class: {}", e))?;

            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("model_path", model_path)
                .map_err(|e| e.to_string())?;

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
