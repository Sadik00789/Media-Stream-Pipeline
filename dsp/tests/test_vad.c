#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "dsp_core.h"
#include "simd_vad.h"

#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "Assertion failed: %s (%s:%d)\n", #cond, __FILE__, __LINE__); \
        abort(); \
    } \
} while (0)

int main(void) {
    printf("[TEST] Running Adaptive SIMD VAD verification suite...\n");
    dsp_init();

    DSP_ALIGNED(32) float silence[512] = {0};
    DSP_ALIGNED(32) float speech[512];
    DSP_ALIGNED(32) float noise[512];

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

    TEST_ASSERT(vad_silence == 0);
    TEST_ASSERT(conf_silence < 0.20f);
    TEST_ASSERT(vad_speech == 1);
    TEST_ASSERT(conf_speech > 0.80f);
    TEST_ASSERT(vad_noise == 0);

    // Null pointer and NaN/Inf resilience
    TEST_ASSERT(dsp_compute_rms_energy_avx2(NULL) == 0.0f);

    float conf_nan = 1.0f;
    uint32_t vad_nan = dsp_vad_decision(NAN, 0.015f, &conf_nan);
    TEST_ASSERT(vad_nan == 0);
    TEST_ASSERT(isfinite(conf_nan) && conf_nan >= 0.0f && conf_nan < 0.5f);

    float conf_inf = 1.0f;
    uint32_t vad_inf = dsp_vad_decision(INFINITY, -1.0f, &conf_inf);
    TEST_ASSERT(vad_inf == 0);
    TEST_ASSERT(isfinite(conf_inf) && conf_inf >= 0.0f && conf_inf < 0.5f);

    // Verify noise floor was not poisoned by NaN/Inf
    float conf_after = 0.0f;
    uint32_t vad_after = dsp_vad_decision(e_speech, 0.015f, &conf_after);
    TEST_ASSERT(vad_after == 1 && conf_after > 0.80f);

    printf("[PASS] Adaptive VAD verification test passed successfully!\n");
    return 0;
}
