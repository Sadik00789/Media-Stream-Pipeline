#include "simd_fft.h"
#include <math.h>

void dsp_fft_512_neon(float* real_in, float* imag_scratch, float* out_spectrogram) {
    dsp_fft_512_avx2(real_in, imag_scratch, out_spectrogram);
}
