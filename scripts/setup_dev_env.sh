#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "============================================================"
echo "  Setting up Media Stream Pipeline Developer Environment"
echo "============================================================"

# 1. Setup Python Virtual Environment
echo -e "\n[1/4] Configuring Python Virtual Environment..."
if [ ! -d "python/.venv" ]; then
    python3 -m venv python/.venv
fi
source python/.venv/bin/activate
pip install --upgrade pip
pip install -e python/ || pip install numpy onnxruntime pytest requests

# 2. Build C DSP Library
echo -e "\n[2/4] Building C11 SIMD DSP Engine..."
cd dsp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cd "$REPO_ROOT"

# 3. Setup Frontend
echo -e "\n[3/4] Installing Node / TypeScript Dependencies..."
cd web
npm install
cd "$REPO_ROOT"

echo -e "\n============================================================"
echo "  Developer Environment Setup Complete!"
echo "  To start the pipeline:"
echo "    Terminal 1: cd crates/ml-worker && cargo run --release"
echo "    Terminal 2: cd cmd/gateway && go run main.go"
echo "    Terminal 3: cd web && npm run dev"
echo "============================================================"
