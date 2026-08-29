#include "dsp_core.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static DspConfig s_config = {
    .vad_threshold = 0.015f,
    .window_type = DSP_WINDOW_HAMMING,
    .sample_rate = 16000,
};

static float s_windowed_scratch[FRAME_SAMPLES] __attribute__((aligned(32)));
static float s_imag_scratch[FRAME_SAMPLES] __attribute__((aligned(32)));
static int s_engine_initialized = 0;

int dsp_init(void) {
    if (s_engine_initialized) {
        return 0;
    }

    dsp_window_init();
    dsp_fft_init();
    dsp_vad_init();

    memset(s_windowed_scratch, 0, sizeof(s_windowed_scratch));
    memset(s_imag_scratch, 0, sizeof(s_imag_scratch));

    s_engine_initialized = 1;
    return 0;
}

void dsp_configure(const DspConfig* config) {
    if (config) {
        s_config = *config;
    }
}

void dsp_process_frame(AudioFramePacket* packet) {
    if (!packet) return;

    if (!s_engine_initialized) {
        if (dsp_init() != 0) return;
    }

    // 1. Copy raw PCM samples into 32-byte aligned scratch buffer
    memcpy(s_windowed_scratch, packet->pcm_data, sizeof(packet->pcm_data));

    // 2. Measure raw frame RMS energy and calculate VAD state & confidence
    float rms_energy = dsp_compute_rms_energy_avx2(packet->pcm_data);
    packet->vad_active = dsp_vad_decision(rms_energy, s_config.vad_threshold, &packet->vad_confidence);

    // 3. Apply configured window function in-place using SIMD
    dsp_apply_window_simd(s_windowed_scratch, s_config.window_type);

    // 4. Compute 512-point Radix-2 FFT and generate normalized decibel spectrogram
    dsp_fft_512_avx2(s_windowed_scratch, s_imag_scratch, packet->spectrogram);

    // 5. Commit DSP processing flag
    packet->flags |= FLAG_DSP_DONE;
}

void dsp_destroy(void) {
    s_engine_initialized = 0;
}
