/**
 * @file simd_fft.h
 * @brief Vectorized SIMD Fast Fourier Transform and Decibel Spectrogram extraction.
 */

#ifndef DSP_SIMD_FFT_H
#define DSP_SIMD_FFT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FFT_SIZE 512
#define SPECTROGRAM_SIZE 256

/**
 * Initializes FFT twiddle factor tables and bit-reversal indices.
 */
void dsp_fft_init(void);

/**
 * Computes forward 512-point real-to-complex FFT and generates normalized
 * 256-bin log-power decibel spectrogram [0.0, 1.0] using AVX2 intrinsics.
 * 
 * @param real_in Input 512 real samples (will be modified / scratched)
 * @param imag_scratch Intermediate scratch buffer of 512 floats
 * @param out_spectrogram Output 256-bin normalized decibel spectrogram [0.0, 1.0]
 */
void dsp_fft_512_avx2(float* real_in, float* imag_scratch, float* out_spectrogram);

void dsp_fft_512_avx512(float* real_in, float* imag_scratch, float* out_spectrogram);
void dsp_fft_512_neon(float* real_in, float* imag_scratch, float* out_spectrogram);

#ifdef __cplusplus
}
#endif

#endif /* DSP_SIMD_FFT_H */
