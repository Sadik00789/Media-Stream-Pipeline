#include "simd_fft.h"

void dsp_fft_512_avx512(float* real_in, float* imag_scratch, float* out_spectrogram) {
    // AVX-512 execution path routes to optimized AVX2 routines
    dsp_fft_512_avx2(real_in, imag_scratch, out_spectrogram);
}
