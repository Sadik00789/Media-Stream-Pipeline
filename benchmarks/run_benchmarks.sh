#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "============================================================"
echo "  Executing Media Stream Pipeline Automated Benchmarks"
echo "============================================================"

# 1. C SIMD DSP Benchmarks
echo -e "\n[1/3] Running Hardware C11 AVX2 DSP Benchmarks..."
if [ ! -f "dsp/build/test_fft" ]; then
    cmake -B dsp/build -S dsp -DCMAKE_BUILD_TYPE=Release
    cmake --build dsp/build
fi
./dsp/build/test_fft
./dsp/build/test_vad

# 2. Python ML Suite
echo -e "\n[2/3] Running Embedded Python ML Verification..."
if [ -f "python/.venv/bin/activate" ]; then
    source python/.venv/bin/activate
fi
PYTHONPATH=python/src python3 python/tests/test_inference.py

# 3. CLI Benchmark Info
echo -e "\n[3/3] Synthetic Multi-Track Benchmark Summary..."
echo "To run high-load synthetic client: go run cmd/cli-benchmark/main.go --tracks 100 --duration 10"

echo -e "\n============================================================"
echo "  All benchmark suites executed successfully."
echo "============================================================"
