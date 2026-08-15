/**
 * @file dsp_core.h
 * @brief Public C-FFI API for the C11 Hardware DSP Engine.
 */

#ifndef DSP_CORE_H
#define DSP_CORE_H

#include "ipc_protocol.h"
#include "windowing.h"
#include "simd_fft.h"
#include "simd_vad.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float vad_threshold;
    DspWindowType window_type;
    uint32_t sample_rate;
} DspConfig;

/**
 * Initializes DSP subsystem, lookup tables, and aligned scratch buffers.
 * Returns 0 on success, non-zero on error.
 */
int dsp_init(void);

/**
 * Configures runtime parameters (VAD threshold, window function, sample rate).
 */
void dsp_configure(const DspConfig* config);

/**
 * In-place processing of single AudioFramePacket:
 * Windowing -> 512-pt FFT -> Decibel Spectrogram -> Adaptive VAD -> Flag commit.
 */
void dsp_process_frame(AudioFramePacket* packet);

/**
 * Cleans up allocated scratch buffers and releases resources.
 */
void dsp_destroy(void);

#ifdef __cplusplus
}
#endif

#endif /* DSP_CORE_H */
