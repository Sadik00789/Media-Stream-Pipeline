# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-11-16

### Added
- **Cross-Platform Shared Memory IPC**: Added Windows file-backed shared memory mapping alongside POSIX `/dev/shm` in both Rust (`ml-worker`) and Go (`internal/ipc`).
- **Full-Stack Automated Test Suites**:
  - C11 (`dsp/tests`): FFT, VAD (`NaN`/`Inf`/silence/tone), and RGB24/NMS vision tests via CMake/CTest.
  - Rust (`crates/`): Unit and integration suites for `tensor-bridge` (`AlignedVec`, FFI ABI layouts, ring-buffer wrap/overrun recovery, PyO3 bindings) and `ml-worker` (`WorkerConfig`, IPC attach, audio/video pipelines).
  - Go (`-race`): Race-tested suites for `internal/config`, `internal/ipc`, `internal/session`, `internal/webrtc`, and `cmd/gateway`.
  - Python (`python/tests`): `pytest` suites covering `StreamingAsrEngine`, `StreamingInferenceEngine`, `VisionEngine`, and tensor sanitization.
  - Web (`web/tests`): Canvas spectrogram, detection overlay, DOM telemetry metrics, and UI control tests (`node --test`).
- **GitHub Actions CI & Release Automation**: Added `.github/workflows/ci.yml`, `.github/workflows/release.yml`, and `.github/dependabot.yml` with multi-OS builds, formatting/lint enforcement (`rustfmt`, `clippy`, `gofmt`, `go vet`), and automated release artifact publishing.
- **Documentation & ABI Alignment**: Updated `VideoFramePacket` ABI size documentation (`922,024` bytes) and synchronized Docker compose/build definitions (`rust:1.85-bookworm`, Nginx `/api/` & `/ws` proxy).

### Fixed
- **C11 DSP & Vision**:
  - Fixed ARM NEON green-channel lane extraction (`vget_low_u16`) in `simd_vision.c`.
  - Eliminated global static scratch buffers in `dsp_core.c` with 32-byte aligned stack-local arrays for thread safety.
  - Guarded `<immintrin.h>`, added null-pointer checks, sanitized non-finite energies in `dsp_vad_decision`, and added MSVC `/arch:AVX2` build flags.
- **Rust (`tensor-bridge` & `ml-worker`)**:
  - Replaced non-atomic `DspEngine` init flag with mutex synchronization and wired `DspEngine::configure`.
  - Prevented zero-size/ZST allocation UB in `AlignedVec` and switched ring-buffer overrun recovery to CAS loops.
  - Eliminated a ~922 KB stack copy in `VideoPipeline::process_packet`, fixed windowed FPS reporting, and forwarded `vad_threshold` and `model_path`.
  - Resolved all Clippy warnings (`-D warnings`) and `rustfmt` diffs.
- **Go Gateway (`cmd/gateway` & `internal/`)**:
  - Added per-writer mutexes on audio/video ring buffers, atomic slot `Flags` clearing before reuse, CAS overrun recovery, and `Fstat`-checked `Ftruncate`.
  - Serialized concurrent WebSocket writes per connection, guarded `Peer.IsActive`, cleaned up peers on SDP failures, and made `/api/telemetry` stateless.
  - Resolved `go vet` `unsafe.Pointer` warnings and applied `gofmt`.
- **Python & Web Frontend**:
  - Skipped 0-byte ONNX placeholder files gracefully and sanitized `NaN`/`Inf` PCM inputs.
  - Synchronized microphone/camera/synthetic-stream UI buttons via `ControlHandle` and prevented conflicting simultaneous streams.
