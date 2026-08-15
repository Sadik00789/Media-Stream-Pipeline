use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[command(name = "ml-worker")]
#[command(about = "Real-time polyglot audio DSP and streaming ML daemon")]
pub struct WorkerConfig {
    /// Name of the POSIX shared memory segment
    #[arg(long, env = "SHM_NAME", default_value = "/media_stream_ring")]
    pub shm_name: String,

    /// Polling interval in microseconds when ring buffer is empty
    #[arg(long, env = "POLL_INTERVAL_US", default_value_t = 200)]
    pub poll_interval_us: u64,

    /// VAD energy threshold
    #[arg(long, env = "VAD_THRESHOLD", default_value_t = 0.015)]
    pub vad_threshold: f32,

    /// Optional path to ONNX model weights
    #[arg(long, env = "MODEL_PATH", default_value = "python/models/silero_vad.onnx")]
    pub model_path: String,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            shm_name: "/media_stream_ring".to_string(),
            poll_interval_us: 200,
            vad_threshold: 0.015,
            model_path: "python/models/silero_vad.onnx".to_string(),
        }
    }
}
