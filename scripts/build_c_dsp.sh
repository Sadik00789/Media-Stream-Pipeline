#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT/dsp"

echo "[Build] Compiling C11 Hardware DSP Engine (libdsp_core.a)..."
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

echo "[Test] Running verification tests..."
./build/test_fft
./build/test_vad

echo "[Success] C DSP static library compiled at $REPO_ROOT/dsp/build/libdsp_core.a"
