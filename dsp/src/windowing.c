#include "windowing.h"
#include <math.h>
#include <immintrin.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define WINDOW_SIZE 512

static float s_hamming_table[WINDOW_SIZE] __attribute__((aligned(32)));
static float s_hanning_table[WINDOW_SIZE] __attribute__((aligned(32)));
static float s_blackman_table[WINDOW_SIZE] __attribute__((aligned(32)));
static int s_initialized = 0;

void dsp_window_init(void) {
    if (s_initialized) return;

    for (int i = 0; i < WINDOW_SIZE; i++) {
        double ratio = (double)i / (double)(WINDOW_SIZE - 1);
        
        // Hamming: 0.54 - 0.46 * cos(2*pi*n/(N-1))
        s_hamming_table[i] = (float)(0.54 - 0.46 * cos(2.0 * M_PI * ratio));
        
        // Hanning: 0.5 * (1 - cos(2*pi*n/(N-1)))
        s_hanning_table[i] = (float)(0.5 * (1.0 - cos(2.0 * M_PI * ratio)));
        
        // Blackman: 0.42 - 0.5 * cos(2*pi*n/(N-1)) + 0.08 * cos(4*pi*n/(N-1))
        s_blackman_table[i] = (float)(0.42 - 0.50 * cos(2.0 * M_PI * ratio) + 0.08 * cos(4.0 * M_PI * ratio));
    }

    s_initialized = 1;
}

void dsp_apply_window_simd(float* data, DspWindowType window_type) {
    if (window_type == DSP_WINDOW_RECTANGULAR) {
        return;
    }

    if (!s_initialized) {
        dsp_window_init();
    }

    const float* win_table = s_hamming_table;
    if (window_type == DSP_WINDOW_HANNING) {
        win_table = s_hanning_table;
    } else if (window_type == DSP_WINDOW_BLACKMAN) {
        win_table = s_blackman_table;
    }

#if defined(__AVX2__)
    for (int i = 0; i < WINDOW_SIZE; i += 8) {
        __m256 v_data = _mm256_loadu_ps(&data[i]);
        __m256 v_win  = _mm256_load_ps(&win_table[i]);
        __m256 v_res  = _mm256_mul_ps(v_data, v_win);
        _mm256_storeu_ps(&data[i], v_res);
    }
#else
    for (int i = 0; i < WINDOW_SIZE; i++) {
        data[i] *= win_table[i];
    }
#endif
}
