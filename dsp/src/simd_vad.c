#include "simd_vad.h"
#include <math.h>
#include <immintrin.h>

#define SAMPLES 512

static float s_noise_floor = 0.005f;
static int s_vad_initialized = 0;

void dsp_vad_init(void) {
    s_noise_floor = 0.005f;
    s_vad_initialized = 1;
}

float dsp_compute_rms_energy_avx2(const float* pcm) {
#if defined(__AVX2__)
    __m256 sum_sq = _mm256_setzero_ps();

    for (int i = 0; i < SAMPLES; i += 8) {
        __m256 v = _mm256_loadu_ps(&pcm[i]);
        sum_sq = _mm256_fmadd_ps(v, v, sum_sq);
    }

    // Horizontal sum of 8 lanes in sum_sq
    __m128 lo = _mm256_castps256_ps128(sum_sq);
    __m128 hi = _mm256_extractf128_ps(sum_sq, 1);
    __m128 sum4 = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehl_ps(sum4, sum4);
    __m128 sum2 = _mm_add_ps(sum4, shuf);
    __m128 shuf2 = _mm_shuffle_ps(sum2, sum2, 1);
    __m128 sum1 = _mm_add_ss(sum2, shuf2);

    float total_sq = _mm_cvtss_f32(sum1);
    return sqrtf(total_sq / (float)SAMPLES);
#else
    float sum_sq = 0.0f;
    for (int i = 0; i < SAMPLES; i++) {
        sum_sq += pcm[i] * pcm[i];
    }
    return sqrtf(sum_sq / (float)SAMPLES);
#endif
}

uint32_t dsp_vad_decision(float energy, float base_threshold, float* out_confidence) {
    if (!s_vad_initialized) {
        dsp_vad_init();
    }

    // Adaptive noise floor tracking (minimum statistics tracker)
    if (energy < s_noise_floor) {
        s_noise_floor = 0.90f * s_noise_floor + 0.10f * energy;
    } else {
        s_noise_floor = 0.999f * s_noise_floor + 0.001f * energy;
    }

    if (s_noise_floor < 0.0001f) s_noise_floor = 0.0001f;
    if (s_noise_floor > 0.1f) s_noise_floor = 0.1f;

    // Dynamic threshold: base threshold + 2.5x estimated noise floor
    float adaptive_thresh = base_threshold + (s_noise_floor * 2.5f);
    if (adaptive_thresh < 0.005f) adaptive_thresh = 0.005f;

    // Sigmoidal confidence curve
    float diff = energy - adaptive_thresh;
    float confidence = 1.0f / (1.0f + expf(-35.0f * diff));

    if (confidence < 0.0f) confidence = 0.0f;
    if (confidence > 1.0f) confidence = 1.0f;

    if (out_confidence) {
        *out_confidence = confidence;
    }

    return (confidence >= 0.50f) ? 1 : 0;
}
