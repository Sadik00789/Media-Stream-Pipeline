/**
 * @file windowing.h
 * @brief Vectorized SIMD audio windowing functions (Hamming, Hanning, Blackman, Rectangular).
 */

#ifndef DSP_WINDOWING_H
#define DSP_WINDOWING_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    DSP_WINDOW_RECTANGULAR = 0,
    DSP_WINDOW_HAMMING = 1,
    DSP_WINDOW_HANNING = 2,
    DSP_WINDOW_BLACKMAN = 3,
} DspWindowType;

/**
 * Initializes window lookup tables for 512 samples.
 */
void dsp_window_init(void);

/**
 * Applies selected window function to input samples in-place using SIMD intrinsics.
 * 
 * @param data Array of 512 float32 PCM samples (32-byte aligned)
 * @param window_type Selected windowing profile
 */
void dsp_apply_window_simd(float* data, DspWindowType window_type);

#ifdef __cplusplus
}
#endif

#endif /* DSP_WINDOWING_H */
