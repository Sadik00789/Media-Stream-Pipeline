#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif
#include "dsp_core.h"
#include "simd_fft.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "Assertion failed: %s (%s:%d)\n", #cond, __FILE__, __LINE__); \
        abort(); \
    } \
} while (0)

static double get_time_us(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1e6 / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
#endif
}

static void test_dsp_core_lifecycle_and_windows(void) {
    TEST_ASSERT(dsp_init() == 0);

    DspWindowType windows[4] = {
        DSP_WINDOW_RECTANGULAR,
        DSP_WINDOW_HAMMING,
        DSP_WINDOW_HANNING,
        DSP_WINDOW_BLACKMAN
    };

    for (int w = 0; w < 4; w++) {
        DspConfig cfg = {
            .vad_threshold = 0.015f,
            .window_type = windows[w],
            .sample_rate = 16000
        };
        dsp_configure(&cfg);

        AudioFramePacket pkt;
        memset(&pkt, 0, sizeof(pkt));
        pkt.sequence_number = (uint64_t)(w + 1);
        pkt.sample_rate = 16000;
        pkt.channels = 1;
        pkt.sample_count = FRAME_SAMPLES;
        pkt.flags = FLAG_INGESTED;

        for (int i = 0; i < FRAME_SAMPLES; i++) {
            pkt.pcm_data[i] = 0.5f * (float)sin(2.0 * M_PI * 1000.0 * (double)i / 16000.0);
        }

        dsp_process_frame(&pkt);

        TEST_ASSERT((pkt.flags & FLAG_DSP_DONE) != 0);
        TEST_ASSERT(pkt.vad_active == 1);
        TEST_ASSERT(pkt.vad_confidence > 0.80f);

        int peak_bin = 0;
        float max_val = -1.0f;
        for (int i = 0; i < SPECTROGRAM_BINS; i++) {
            TEST_ASSERT(isfinite(pkt.spectrogram[i]));
            TEST_ASSERT(pkt.spectrogram[i] >= 0.0f && pkt.spectrogram[i] <= 1.0f);
            if (pkt.spectrogram[i] > max_val) {
                max_val = pkt.spectrogram[i];
                peak_bin = i;
            }
        }
        TEST_ASSERT(peak_bin == 32);
        TEST_ASSERT(max_val > 0.70f);
    }

    // Null pointer safety checks
    dsp_process_frame(NULL);
    dsp_apply_window_simd(NULL, DSP_WINDOW_HAMMING);
    dsp_fft_512_avx2(NULL, NULL, NULL);
    dsp_configure(NULL);
    dsp_destroy();
    TEST_ASSERT(dsp_init() == 0);
}

int main(void) {
    printf("[TEST] Running SIMD Decibel FFT verification suite...\n");
    dsp_init();

    DSP_ALIGNED(32) float pcm[512];
    DSP_ALIGNED(32) float imag[512];
    DSP_ALIGNED(32) float spec[256];

    // Generate 1000 Hz tone at 16000 Hz sample rate (Bin 32)
    for (int i = 0; i < 512; i++) {
        pcm[i] = 0.5f * (float)sin(2.0 * M_PI * 1000.0 * (double)i / 16000.0);
    }

    dsp_fft_512_avx2(pcm, imag, spec);

    // Find peak frequency bin
    int peak_bin = 0;
    float max_val = 0.0f;
    for (int i = 0; i < 256; i++) {
        TEST_ASSERT(spec[i] >= 0.0f && spec[i] <= 1.0f);
        if (spec[i] > max_val) {
            max_val = spec[i];
            peak_bin = i;
        }
    }

    printf("  -> Expected Peak Bin: 32, Detected Peak Bin: %d (Norm dB Val: %.4f)\n", peak_bin, max_val);
    TEST_ASSERT(peak_bin == 32);
    TEST_ASSERT(max_val > 0.80f);

    test_dsp_core_lifecycle_and_windows();
    printf("  -> DSP lifecycle, all 4 window types, and NULL-pointer guards verified.\n");

    // Micro-benchmark
    int iterations = 10000;
    double start_us = get_time_us();

    for (int i = 0; i < iterations; i++) {
        dsp_fft_512_avx2(pcm, imag, spec);
    }

    double end_us = get_time_us();
    double us_per_call = (end_us - start_us) / (double)iterations;

    printf("  -> Benchmark: %.2f us per 512-pt FFT + dB Spectrogram (%d iterations)\n", us_per_call, iterations);
    printf("[PASS] Decibel FFT verification test passed successfully!\n");
    return 0;
}
