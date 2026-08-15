#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "dsp_core.h"
#include "simd_vad.h"

int main(void) {
    printf("[TEST] Running Adaptive SIMD VAD verification suite...\n");
    dsp_init();

    float silence[512] __attribute__((aligned(32))) = {0};
    float speech[512] __attribute__((aligned(32)));
    float noise[512] __attribute__((aligned(32)));

    // Generate speech signal (mixed sine waves)
    for (int i = 0; i < 512; i++) {
        speech[i] = 0.4f * (float)sin((double)i * 0.1) + 0.3f * (float)cos((double)i * 0.05);
        noise[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.005f; // low background noise
    }

    float conf_silence = 0.0f;
    float conf_speech = 0.0f;
    float conf_noise = 0.0f;

    float e_silence = dsp_compute_rms_energy_avx2(silence);
    uint32_t vad_silence = dsp_vad_decision(e_silence, 0.015f, &conf_silence);

    float e_speech = dsp_compute_rms_energy_avx2(speech);
    uint32_t vad_speech = dsp_vad_decision(e_speech, 0.015f, &conf_speech);

    float e_noise = dsp_compute_rms_energy_avx2(noise);
    uint32_t vad_noise = dsp_vad_decision(e_noise, 0.015f, &conf_noise);

    printf("  -> Silent Frame: VAD Active = %u, Confidence = %.4f\n", vad_silence, conf_silence);
    printf("  -> Active Speech Frame: VAD Active = %u, Confidence = %.4f\n", vad_speech, conf_speech);
    printf("  -> Low Noise Frame: VAD Active = %u, Confidence = %.4f\n", vad_noise, conf_noise);

    assert(vad_silence == 0);
    assert(conf_silence < 0.20f);
    assert(vad_speech == 1);
    assert(conf_speech > 0.80f);
    assert(vad_noise == 0);

    printf("[PASS] Adaptive VAD verification test passed successfully!\n");
    return 0;
}
