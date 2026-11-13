use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[command(name = "ml-worker")]
#[command(about = "Real-time polyglot dual-track audio DSP and vision ML daemon")]
pub struct WorkerConfig {
    /// Name of the POSIX audio shared memory segment
    #[arg(long, env = "SHM_NAME", default_value = "/media_stream_ring")]
    pub shm_name: String,

    /// Name of the POSIX video shared memory segment
    #[arg(long, env = "VIDEO_SHM_NAME", default_value = "/video_stream_ring")]
    pub video_shm_name: String,

    /// Polling interval in microseconds when audio ring buffer is empty
    #[arg(long, env = "POLL_INTERVAL_US", default_value_t = 200)]
    pub poll_interval_us: u64,

    /// Polling interval in microseconds when video ring buffer is empty
    #[arg(long, env = "VIDEO_POLL_INTERVAL_US", default_value_t = 1000)]
    pub video_poll_interval_us: u64,

    /// VAD energy threshold
    #[arg(long, env = "VAD_THRESHOLD", default_value_t = 0.015)]
    pub vad_threshold: f32,

    /// Optional path to Audio ONNX model weights
    #[arg(long, env = "MODEL_PATH", default_value = "python/models/silero_vad.onnx")]
    pub model_path: String,

    /// Optional path to Vision ONNX model weights
    #[arg(long, env = "VISION_MODEL_PATH", default_value = "python/models/ultraface.onnx")]
    pub vision_model_path: String,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            shm_name: "/media_stream_ring".to_string(),
            video_shm_name: "/video_stream_ring".to_string(),
            poll_interval_us: 200,
            video_poll_interval_us: 1000,
            vad_threshold: 0.015,
            model_path: "python/models/silero_vad.onnx".to_string(),
            vision_model_path: "python/models/ultraface.onnx".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_config_defaults_and_cli_override() {
        let def = WorkerConfig::default();
        assert_eq!(def.shm_name, "/media_stream_ring");
        assert_eq!(def.video_shm_name, "/video_stream_ring");
        assert!((def.vad_threshold - 0.015).abs() < 1e-6);

        let parsed = WorkerConfig::try_parse_from([
            "ml-worker",
            "--shm-name",
            "/custom_audio",
            "--video-shm-name",
            "/custom_video",
            "--vad-threshold",
            "0.045",
            "--poll-interval-us",
            "500",
        ])
        .expect("Failed to parse CLI args");

        assert_eq!(parsed.shm_name, "/custom_audio");
        assert_eq!(parsed.video_shm_name, "/custom_video");
        assert!((parsed.vad_threshold - 0.045).abs() < 1e-6);
        assert_eq!(parsed.poll_interval_us, 500);
    }
}
