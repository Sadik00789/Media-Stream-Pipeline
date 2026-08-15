/**
 * @file simd_vad.h
 * @brief Vectorized Voice Activity Detection with Adaptive Noise Floor Tracking.
 */

#ifndef DSP_SIMD_VAD_H
#define DSP_SIMD_VAD_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initializes adaptive VAD state and minimum statistics energy trackers.
 */
void dsp_vad_init(void);

/**
 * Computes root-mean-square (RMS) energy of 512 PCM audio samples using AVX2 FMA.
 * 
 * @param pcm 512 float32 PCM samples (32-byte aligned)
 * @return Computed RMS energy float value
 */
float dsp_compute_rms_energy_avx2(const float* pcm);

/**
 * Updates adaptive noise floor estimate and makes speech detection decision.
 * 
 * @param energy Measured frame RMS energy
 * @param base_threshold Base sensitivity threshold (typically 0.015)
 * @param out_confidence Output pointer for confidence score [0.0, 1.0]
 * @return 1 if speech detected, 0 otherwise
 */
uint32_t dsp_vad_decision(float energy, float base_threshold, float* out_confidence);

#ifdef __cplusplus
}
#endif

#endif /* DSP_SIMD_VAD_H */
