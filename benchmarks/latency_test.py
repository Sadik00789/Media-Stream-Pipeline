#!/usr/bin/env python3
"""
End-to-End Latency Benchmark for Media Stream Pipeline.
Pushes synthetic 16kHz audio frames to Gateway and measures processing latency.
"""

import time
import requests
import numpy as np

GATEWAY_URL = "http://localhost:8080"
SAMPLE_RATE = 16000
FRAME_SIZE = 512
NUM_FRAMES = 1000


def run_latency_benchmark():
    print("=" * 60)
    print("  Media Stream Pipeline - E2E Latency Benchmark")
    print("=" * 60)
    print(f"Target:        {GATEWAY_URL}")
    print(f"Sample Rate:   {SAMPLE_RATE} Hz")
    print(f"Frame Size:    {FRAME_SIZE} samples (32 ms window)")
    print(f"Total Frames:  {NUM_FRAMES}")
    print("-" * 60)

    # 1. Generate 1000 Hz pure sine wave
    t = np.linspace(0, FRAME_SIZE / SAMPLE_RATE, FRAME_SIZE, endpoint=False, dtype=np.float32)
    tone_frame = (0.5 * np.sin(2 * np.pi * 1000.0 * t)).tolist()

    latencies_ms = []

    # Warmup
    for _ in range(20):
        try:
            requests.post(f"{GATEWAY_URL}/api/pcm", json={"samples": tone_frame}, timeout=1.0)
        except Exception:
            pass

    print("Executing benchmark stream...")
    start_total = time.time()

    for i in range(NUM_FRAMES):
        t0 = time.perf_counter()
        try:
            resp = requests.post(f"{GATEWAY_URL}/api/pcm", json={"samples": tone_frame}, timeout=1.0)
            t1 = time.perf_counter()
            if resp.status_code == 202:
                latencies_ms.append((t1 - t0) * 1000.0)
        except Exception as e:
            print(f"Error on frame {i}: {e}")
            break

        # Simulate real-time 32ms audio frame rate pacing
        time.sleep(0.001)

    total_time = time.time() - start_total

    if not latencies_ms:
        print("\n[Error] No successful frame responses received from Gateway.")
        return

    latencies = np.array(latencies_ms)
    print("\n" + "=" * 60)
    print("  Latency Distribution Results (ms)")
    print("=" * 60)
    print(f"Min Latency:   {np.min(latencies):.3f} ms")
    print(f"P50 (Median):  {np.percentile(latencies, 50):.3f} ms")
    print(f"P95:           {np.percentile(latencies, 95):.3f} ms")
    print(f"P99:           {np.percentile(latencies, 99):.3f} ms")
    print(f"Max Latency:   {np.max(latencies):.3f} ms")
    print(f"Throughput:    {len(latencies) / total_time:.2f} frames/sec")
    print("=" * 60)


if __name__ == "__main__":
    run_latency_benchmark()
