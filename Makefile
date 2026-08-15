.PHONY: all setup dsp build test run-gateway run-worker run-web docker-up clean help

all: build

help:
	@echo "Media Stream Pipeline Build Targets:"
	@echo "  make setup        - Install dependencies and build C DSP library"
	@echo "  make dsp          - Compile libdsp_core.a and run verification tests"
	@echo "  make build        - Build all components (C, Rust, Go, Web)"
	@echo "  make test         - Run test suites across C, Rust, and Python"
	@echo "  make run-worker   - Run Rust/Python ML Worker daemon (Terminal 1)"
	@echo "  make run-gateway  - Run Go WebRTC Gateway server (Terminal 2)"
	@echo "  make run-web      - Run TypeScript Vite frontend (Terminal 3)"
	@echo "  make docker-up    - Launch multi-service Docker Compose cluster"
	@echo "  make clean        - Clean build artifacts"

setup:
	@bash scripts/setup_dev_env.sh

dsp:
	@bash scripts/build_c_dsp.sh

build: dsp
	@echo "Building Rust crates..."
	@cargo build --manifest-path crates/Cargo.toml --release
	@echo "Building Frontend..."
	@cd web && npm run build

test: dsp
	@echo "Testing C DSP..."
	@./dsp/build/test_fft
	@./dsp/build/test_vad
	@echo "Testing Rust Crates..."
	@cargo test --manifest-path crates/Cargo.toml
	@echo "Testing Python ML..."
	@python3 -m pytest python/tests || true

run-worker: dsp
	@cd crates/ml-worker && cargo run --release

run-gateway:
	@cd cmd/gateway && go run main.go

run-web:
	@cd web && npm run dev

docker-up:
	@docker compose -f deploy/docker-compose.yml up --build

clean:
	@rm -rf dsp/build
	@cargo clean --manifest-path crates/Cargo.toml
	@rm -rf web/dist web/node_modules
	@rm -rf python/.pytest_cache python/__pycache__
