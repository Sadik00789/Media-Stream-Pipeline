#include "simd_fft.h"
#include <math.h>
#include <immintrin.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static float s_cos_table[FFT_SIZE / 2] __attribute__((aligned(32)));
static float s_sin_table[FFT_SIZE / 2] __attribute__((aligned(32)));
static uint16_t s_bit_rev[FFT_SIZE] __attribute__((aligned(32)));
static int s_fft_initialized = 0;

static uint16_t reverse_bits(uint16_t x, int bits) {
    uint16_t res = 0;
    for (int i = 0; i < bits; i++) {
        if (x & (1 << i)) {
            res |= (1 << (bits - 1 - i));
        }
    }
    return res;
}

void dsp_fft_init(void) {
    if (s_fft_initialized) return;

    for (int i = 0; i < FFT_SIZE / 2; i++) {
        double angle = -2.0 * M_PI * (double)i / (double)FFT_SIZE;
        s_cos_table[i] = (float)cos(angle);
        s_sin_table[i] = (float)sin(angle);
    }

    for (int i = 0; i < FFT_SIZE; i++) {
        s_bit_rev[i] = reverse_bits((uint16_t)i, 9); // log2(512) = 9
    }

    s_fft_initialized = 1;
}

#if defined(__AVX2__)

// Fast AVX2 vector natural logarithm approximation
static inline __m256 fast_log_avx2(__m256 x) {
    // log(x) ~ ln(x)
    // Extract exponent and mantissa using IEEE 754 float representation
    __m256i i_val = _mm256_castps_si256(x);
    __m256i exp_i = _mm256_sub_epi32(_mm256_srli_epi32(i_val, 23), _mm256_set1_epi32(127));
    __m256 exp_f  = _mm256_cvtepi32_ps(exp_i);

    // Mantissa normalized to [1.0, 2.0)
    __m256i mant_i = _mm256_or_si256(_mm256_and_si256(i_val, _mm256_set1_epi32(0x007FFFFF)), _mm256_set1_epi32(0x3F800000));
    __m256 m = _mm256_castsi256_ps(mant_i);

    // Polynomial approximation of ln(m) on [1, 2]: -1.7417939 + m * (2.8212026 + m * (-1.4699568 + m * 0.3905184))
    __m256 p = _mm256_set1_ps(0.3905184f);
    p = _mm256_fmadd_ps(p, m, _mm256_set1_ps(-1.4699568f));
    p = _mm256_fmadd_ps(p, m, _mm256_set1_ps(2.8212026f));
    p = _mm256_fmadd_ps(p, m, _mm256_set1_ps(-1.7417939f));

    // ln(x) = ln(2) * exp + ln(m)
    __m256 ln2 = _mm256_set1_ps(0.69314718056f);
    return _mm256_fmadd_ps(exp_f, ln2, p);
}

#endif

void dsp_fft_512_avx2(float* real_in, float* imag_scratch, float* out_spectrogram) {
    if (!s_fft_initialized) {
        dsp_fft_init();
    }

    // 1. In-place bit-reversal permutation into temporary buffer
    float temp_re[FFT_SIZE] __attribute__((aligned(32)));
    float temp_im[FFT_SIZE] __attribute__((aligned(32)));

    for (int i = 0; i < FFT_SIZE; i++) {
        int rev = s_bit_rev[i];
        temp_re[rev] = real_in[i];
        temp_im[rev] = 0.0f;
    }

    // 2. Radix-2 Cooley-Tukey Butterfly Stages
    for (int stage = 1; stage <= 9; stage++) {
        int m = 1 << stage;
        int m2 = m >> 1;
        int step = FFT_SIZE / m;

        for (int k = 0; k < FFT_SIZE; k += m) {
            for (int j = 0; j < m2; j++) {
                int twiddle_idx = j * step;
                float wr = s_cos_table[twiddle_idx];
                float wi = s_sin_table[twiddle_idx];

                int u_idx = k + j;
                int t_idx = u_idx + m2;

                float tr = wr * temp_re[t_idx] - wi * temp_im[t_idx];
                float ti = wr * temp_im[t_idx] + wi * temp_re[t_idx];

                temp_re[t_idx] = temp_re[u_idx] - tr;
                temp_im[t_idx] = temp_im[u_idx] - ti;

                temp_re[u_idx] = temp_re[u_idx] + tr;
                temp_im[u_idx] = temp_im[u_idx] + ti;
            }
        }
    }

    // 3. Decibel-Scaled Log-Power Spectrogram Normalization
    // dB = 20 * log10(magnitude + 1e-6)
    // Normalized [-80 dB, 0 dB] -> [0.0, 1.0]
#if defined(__AVX2__)
    __m256 scale_inv = _mm256_set1_ps(1.0f / 256.0f);
    __m256 eps       = _mm256_set1_ps(1e-6f);
    __m256 log10_e   = _mm256_set1_ps(0.434294481903f); // log10(e) for ln -> log10 conversion
    __m256 twenty    = _mm256_set1_ps(20.0f);
    __m256 eighty    = _mm256_set1_ps(80.0f);
    __m256 norm_div  = _mm256_set1_ps(1.0f / 80.0f);
    __m256 zero      = _mm256_setzero_ps();
    __m256 one       = _mm256_set1_ps(1.0f);

    for (int i = 0; i < SPECTROGRAM_SIZE; i += 8) {
        __m256 r = _mm256_load_ps(&temp_re[i]);
        __m256 im = _mm256_load_ps(&temp_im[i]);

        // Power: r^2 + im^2
        __m256 pwr = _mm256_add_ps(_mm256_mul_ps(r, r), _mm256_mul_ps(im, im));
        __m256 mag = _mm256_mul_ps(_mm256_sqrt_ps(pwr), scale_inv);

        // mag + eps
        __m256 mag_eps = _mm256_add_ps(mag, eps);

        // ln(mag + eps)
        __m256 ln_val = fast_log_avx2(mag_eps);

        // dB = 20 * (ln_val * log10(e))
        __m256 log10_val = _mm256_mul_ps(ln_val, log10_e);
        __m256 db = _mm256_mul_ps(twenty, log10_val);

        // normalized = (dB + 80.0) / 80.0
        __m256 norm = _mm256_mul_ps(_mm256_add_ps(db, eighty), norm_div);

        // clamp [0.0, 1.0]
        norm = _mm256_max_ps(zero, _mm256_min_ps(one, norm));

        _mm256_storeu_ps(&out_spectrogram[i], norm);
    }
#else
    for (int i = 0; i < SPECTROGRAM_SIZE; i++) {
        float r = temp_re[i];
        float im = temp_im[i];
        float mag = sqrtf(r * r + im * im) / 256.0f;
        float db = 20.0f * log10f(mag + 1e-6f);
        float norm = (db + 80.0f) / 80.0f;
        if (norm < 0.0f) norm = 0.0f;
        if (norm > 1.0f) norm = 1.0f;
        out_spectrogram[i] = norm;
    }
#endif
}
