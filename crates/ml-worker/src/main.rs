mod config;
mod ipc;
mod pipeline;
mod py_runtime;

use clap::Parser;
use config::WorkerConfig;
use ipc::ShmContext;
use pipeline::AudioPipeline;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
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
    info!("Starting ML Worker Daemon with config: {:?}", config);

    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
        info!("Received shutdown signal. Exiting...");
    })
    .ok();

    // 1. Initialize Audio Processing Pipeline (DSP + PyO3)
    let mut pipeline = AudioPipeline::new(&config.model_path)
        .map_err(|e| format!("Failed to initialize audio pipeline: {}", e))?;

    // 2. Attach to POSIX Shared Memory Ring Buffer (retry loop until Gateway initializes SHM)
    info!("Waiting for Shared Memory segment '{}'...", config.shm_name);
    let (_shm_ctx, mut shm_reader) = loop {
        if !running.load(Ordering::SeqCst) {
            return Ok(());
        }
        match ShmContext::attach(&config.shm_name) {
            Ok(res) => {
                info!("Successfully attached to POSIX Shared Memory segment!");
                break res;
            }
            Err(_) => {
                std::thread::sleep(Duration::from_millis(250));
            }
        }
    };

    let poll_sleep = Duration::from_micros(config.poll_interval_us);
    let mut processed_frames: u64 = 0;
    let mut last_stats_time = Instant::now();

    info!("ML Worker hot-path polling loop started.");

    while running.load(Ordering::SeqCst) {
        if let Some(packet) = shm_reader.poll_next_packet() {
            let start = Instant::now();
            let seq = packet.sequence_number;

            // Run SIMD DSP and embedded ML inference
            pipeline.process_packet(packet);

            // Advance atomic read cursor
            shm_reader.advance_read_head();
            processed_frames += 1;

            let elapsed_us = start.elapsed().as_micros();
            if elapsed_us > 20000 {
                warn!("Frame {} processing latency exceeded 20ms: {} µs", seq, elapsed_us);
            }
        } else {
            // Buffer empty, yield CPU briefly
            std::thread::sleep(poll_sleep);
        }

        // Periodic throughput and telemetry logging
        if last_stats_time.elapsed() >= Duration::from_secs(5) {
            if processed_frames > 0 {
                let fps = (processed_frames as f64) / last_stats_time.elapsed().as_secs_f64();
                info!("Throughput: {:.2} frames/sec (Total: {})", fps, processed_frames);
            }
            last_stats_time = Instant::now();
        }
    }

    info!("ML Worker shut down cleanly.");
    Ok(())
}
