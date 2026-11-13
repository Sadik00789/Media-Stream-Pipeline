mod config;
mod ipc;
mod pipeline;
mod py_runtime;

use clap::Parser;
use config::WorkerConfig;
use ipc::{ShmContext, VideoShmContext};
use pipeline::{AudioPipeline, VideoPipeline};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use tracing::{info, warn};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let config = WorkerConfig::parse();
    info!("Starting Dual-Track Audio + Video ML Worker Daemon with config: {:?}", config);

    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
        info!("Received shutdown signal. Exiting...");
    })
    .ok();

    // 1. Audio Processing Worker Thread
    let running_audio = running.clone();
    let config_audio = config.clone();
    let audio_handle = thread::spawn(move || {
        info!("Initializing Audio Processing Pipeline (C11 SIMD + PyO3)...");
        let mut audio_pipeline =
            match AudioPipeline::new_with_config(&config_audio.model_path, config_audio.vad_threshold) {
                Ok(p) => p,
                Err(e) => {
                    warn!("Failed to initialize audio pipeline: {}", e);
                    return;
                }
            };

        info!("Waiting for Audio Shared Memory segment '{}'...", config_audio.shm_name);
        let (_shm_ctx, mut shm_reader) = loop {
            if !running_audio.load(Ordering::SeqCst) {
                return;
            }
            match ShmContext::attach(&config_audio.shm_name) {
                Ok(res) => {
                    info!("Successfully attached to Audio Shared Memory segment!");
                    break res;
                }
                Err(_) => {
                    thread::sleep(Duration::from_millis(250));
                }
            }
        };

        let poll_sleep = Duration::from_micros(config_audio.poll_interval_us);
        let mut processed_frames: u64 = 0;
        let mut last_processed_frames: u64 = 0;
        let mut last_stats_time = Instant::now();

        info!("Audio ML Worker hot-path polling loop started.");

        while running_audio.load(Ordering::SeqCst) {
            if let Some(packet) = shm_reader.poll_next_packet() {
                let start = Instant::now();
                let seq = packet.sequence_number;

                // Run SIMD DSP and embedded ML inference
                audio_pipeline.process_packet(packet);

                // Advance atomic read cursor
                shm_reader.advance_read_head();
                processed_frames += 1;

                let elapsed_us = start.elapsed().as_micros();
                if elapsed_us > 20000 {
                    warn!("Audio Frame {} processing latency exceeded 20ms: {} µs", seq, elapsed_us);
                }
            } else {
                thread::sleep(poll_sleep);
            }

            if last_stats_time.elapsed() >= Duration::from_secs(5) {
                let delta_frames = processed_frames - last_processed_frames;
                if delta_frames > 0 {
                    let fps = (delta_frames as f64) / last_stats_time.elapsed().as_secs_f64();
                    info!("Audio Throughput: {:.2} frames/sec (Total: {})", fps, processed_frames);
                    last_processed_frames = processed_frames;
                }
                last_stats_time = Instant::now();
            }
        }
        info!("Audio ML Worker thread stopped.");
    });

    // 2. Video Processing Worker Thread
    let running_video = running.clone();
    let config_video = config.clone();
    let video_handle = thread::spawn(move || {
        info!("Initializing Video Processing Pipeline (PyO3 Vision + SIMD NMS)...");
        let mut video_pipeline = match VideoPipeline::new(&config_video.vision_model_path) {
            Ok(p) => p,
            Err(e) => {
                warn!("Failed to initialize video pipeline: {}", e);
                return;
            }
        };

        info!("Waiting for Video Shared Memory segment '{}'...", config_video.video_shm_name);
        let (_video_shm_ctx, mut video_shm_reader) = loop {
            if !running_video.load(Ordering::SeqCst) {
                return;
            }
            match VideoShmContext::attach(&config_video.video_shm_name) {
                Ok(res) => {
                    info!("Successfully attached to Video Shared Memory segment!");
                    break res;
                }
                Err(_) => {
                    thread::sleep(Duration::from_millis(250));
                }
            }
        };

        let poll_sleep = Duration::from_micros(config_video.video_poll_interval_us);
        let mut processed_frames: u64 = 0;
        let mut last_processed_frames: u64 = 0;
        let mut last_stats_time = Instant::now();

        info!("Video ML Worker hot-path polling loop started.");

        while running_video.load(Ordering::SeqCst) {
            if let Some(packet) = video_shm_reader.poll_next_packet() {
                let start = Instant::now();
                let seq = packet.sequence_number;

                // Run Python Vision inference and SIMD NMS filtering
                video_pipeline.process_packet(packet);

                // Advance atomic read cursor
                video_shm_reader.advance_read_head();
                processed_frames += 1;

                let elapsed_us = start.elapsed().as_micros();
                if elapsed_us > 33000 {
                    warn!("Video Frame {} processing latency exceeded 33ms (30 FPS budget): {} µs", seq, elapsed_us);
                }
            } else {
                thread::sleep(poll_sleep);
            }

            if last_stats_time.elapsed() >= Duration::from_secs(5) {
                let delta_frames = processed_frames - last_processed_frames;
                if delta_frames > 0 {
                    let fps = (delta_frames as f64) / last_stats_time.elapsed().as_secs_f64();
                    info!("Video Throughput: {:.2} frames/sec (Total: {})", fps, processed_frames);
                    last_processed_frames = processed_frames;
                }
                last_stats_time = Instant::now();
            }
        }
        info!("Video ML Worker thread stopped.");
    });

    let _ = audio_handle.join();
    let _ = video_handle.join();

    info!("ML Worker daemon shut down cleanly.");
    Ok(())
}
