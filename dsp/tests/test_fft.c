#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include "dsp_core.h"
#include "simd_fft.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main(void) {
    printf("[TEST] Running SIMD Decibel FFT verification suite...\n");
    dsp_init();

    float pcm[512] __attribute__((aligned(32)));
    float imag[512] __attribute__((aligned(32)));
    float spec[256] __attribute__((aligned(32)));

    // Generate 1000 Hz tone at 16000 Hz sample rate (Bin 32)
    for (int i = 0; i < 512; i++) {
        pcm[i] = 0.5f * (float)sin(2.0 * M_PI * 1000.0 * (double)i / 16000.0);
    }

    dsp_fft_512_avx2(pcm, imag, spec);

    // Find peak frequency bin
    int peak_bin = 0;
    float max_val = 0.0f;
    for (int i = 0; i < 256; i++) {
        assert(spec[i] >= 0.0f && spec[i] <= 1.0f);
        if (spec[i] > max_val) {
            max_val = spec[i];
            peak_bin = i;
        }
    }

    printf("  -> Expected Peak Bin: 32, Detected Peak Bin: %d (Norm dB Val: %.4f)\n", peak_bin, max_val);
    assert(peak_bin == 32);
    assert(max_val > 0.80f);

    // Micro-benchmark
    int iterations = 10000;
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < iterations; i++) {
        dsp_fft_512_avx2(pcm, imag, spec);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double total_us = (double)(end.tv_sec - start.tv_sec) * 1e6 + (double)(end.tv_nsec - start.tv_nsec) / 1e3;
    double us_per_call = total_us / (double)iterations;

    printf("  -> Benchmark: %.2f µs per 512-pt FFT + dB Spectrogram (%d iterations)\n", us_per_call, iterations);
    printf("[PASS] Decibel FFT verification test passed successfully!\n");
    return 0;
}
