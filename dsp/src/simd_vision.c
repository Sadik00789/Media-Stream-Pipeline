/**
 * @file simd_vision.c
 * @brief High-performance C11 SIMD Vision Kernels (AVX2 / NEON / Scalar).
 */

#include "simd_vision.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

void simd_rgb24_to_planar_f32(const uint8_t* rgb_in, float* planar_out, int width, int height) {
    if (!rgb_in || !planar_out || width <= 0 || height <= 0) {
        return;
    }

    const int total_pixels = width * height;
    float* r_plane = planar_out;
    float* g_plane = planar_out + total_pixels;
    float* b_plane = planar_out + (2 * total_pixels);
    const float scale = 1.0f / 255.0f;

    int i = 0;

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    float32x4_t vscale = vdupq_n_f32(scale);
    for (; i + 16 <= total_pixels; i += 16) {
        // vld3q_u8 deinterleaves 16 RGB pixels directly into 3 separate channel vectors
        uint8x16x3_t rgb = vld3q_u8(&rgb_in[i * 3]);

        // Process R channel
        uint16x8_t r_u16_low = vmovl_u8(vget_low_u8(rgb.val[0]));
        uint16x8_t r_u16_high = vmovl_u8(vget_high_u8(rgb.val[0]));
        
        vst1q_f32(&r_plane[i + 0], vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(r_u16_low))), vscale));
        vst1q_f32(&r_plane[i + 4], vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(r_u16_low))), vscale));
        vst1q_f32(&r_plane[i + 8], vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(r_u16_high))), vscale));
        vst1q_f32(&r_plane[i + 12], vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(r_u16_high))), vscale));

        // Process G channel
        uint16x8_t g_u16_low = vmovl_u8(vget_low_u8(rgb.val[1]));
        uint16x8_t g_u16_high = vmovl_u8(vget_high_u8(rgb.val[1]));
        
        vst1q_f32(&g_plane[i + 0], vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(g_u16_low))), vscale));
        vst1q_f32(&g_plane[i + 4], vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(g_u16_low))), vscale));
        vst1q_f32(&g_plane[i + 8], vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(g_u16_high))), vscale));
        vst1q_f32(&g_plane[i + 12], vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(g_u16_high))), vscale));

        // Process B channel
        uint16x8_t b_u16_low = vmovl_u8(vget_low_u8(rgb.val[2]));
        uint16x8_t b_u16_high = vmovl_u8(vget_high_u8(rgb.val[2]));
        
        vst1q_f32(&b_plane[i + 0], vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(b_u16_low))), vscale));
        vst1q_f32(&b_plane[i + 4], vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(b_u16_low))), vscale));
        vst1q_f32(&b_plane[i + 8], vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(b_u16_high))), vscale));
        vst1q_f32(&b_plane[i + 12], vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(b_u16_high))), vscale));
    }
#elif defined(__AVX2__)
    __m256 vscale = _mm256_set1_ps(scale);

    for (; i + 8 <= total_pixels; i += 8) {
        // Extract 8 RGB triplets (24 bytes) into separate float arrays
        float r_tmp[8], g_tmp[8], b_tmp[8];
        for (int k = 0; k < 8; k++) {
            const int in_idx = (i + k) * 3;
            r_tmp[k] = (float)rgb_in[in_idx + 0];
            g_tmp[k] = (float)rgb_in[in_idx + 1];
            b_tmp[k] = (float)rgb_in[in_idx + 2];
        }

        __m256 vr = _mm256_mul_ps(_mm256_loadu_ps(r_tmp), vscale);
        __m256 vg = _mm256_mul_ps(_mm256_loadu_ps(g_tmp), vscale);
        __m256 vb = _mm256_mul_ps(_mm256_loadu_ps(b_tmp), vscale);

        _mm256_storeu_ps(&r_plane[i], vr);
        _mm256_storeu_ps(&g_plane[i], vg);
        _mm256_storeu_ps(&b_plane[i], vb);
    }
#endif

    // Remainder scalar loop
    for (; i < total_pixels; i++) {
        const int in_idx = i * 3;
        r_plane[i] = (float)rgb_in[in_idx + 0] * scale;
        g_plane[i] = (float)rgb_in[in_idx + 1] * scale;
        b_plane[i] = (float)rgb_in[in_idx + 2] * scale;
    }
}

static inline float compute_iou(const DetectionBox* a, const DetectionBox* b) {
    float ymin = a->ymin > b->ymin ? a->ymin : b->ymin;
    float xmin = a->xmin > b->xmin ? a->xmin : b->xmin;
    float ymax = a->ymax < b->ymax ? a->ymax : b->ymax;
    float xmax = a->xmax < b->xmax ? a->xmax : b->xmax;

    float inter_h = (ymax - ymin > 0.0f) ? (ymax - ymin) : 0.0f;
    float inter_w = (xmax - xmin > 0.0f) ? (xmax - xmin) : 0.0f;
    float inter_area = inter_h * inter_w;

    float area_a = (a->ymax - a->ymin) * (a->xmax - a->xmin);
    float area_b = (b->ymax - b->ymin) * (b->xmax - b->xmin);

    if (area_a <= 0.0f || area_b <= 0.0f) {
        return 0.0f;
    }

    float union_area = area_a + area_b - inter_area;
    if (union_area <= 0.0f) {
        return 0.0f;
    }

    return inter_area / union_area;
}

typedef struct {
    int index;
    float confidence;
} BoxScore;

static int compare_box_scores(const void* a, const void* b) {
    const BoxScore* sa = (const BoxScore*)a;
    const BoxScore* sb = (const BoxScore*)b;
    if (sb->confidence > sa->confidence) return 1;
    if (sb->confidence < sa->confidence) return -1;
    return 0;
}

uint32_t simd_fast_box_nms(
    const DetectionBox* boxes_in,
    uint32_t num_boxes,
    float iou_threshold,
    DetectionBox* boxes_out,
    uint32_t max_out
) {
    if (!boxes_in || !boxes_out || num_boxes == 0 || max_out == 0) {
        return 0;
    }

    if (num_boxes > 256) {
        num_boxes = 256;
    }

    BoxScore scores[256];
    uint8_t suppressed[256];
    memset(suppressed, 0, sizeof(suppressed));

    for (uint32_t i = 0; i < num_boxes; i++) {
        scores[i].index = (int)i;
        scores[i].confidence = boxes_in[i].confidence;
    }

    qsort(scores, num_boxes, sizeof(BoxScore), compare_box_scores);

    uint32_t count = 0;
    for (uint32_t i = 0; i < num_boxes && count < max_out; i++) {
        int orig_idx = scores[i].index;
        if (suppressed[orig_idx]) {
            continue;
        }

        boxes_out[count++] = boxes_in[orig_idx];

        for (uint32_t j = i + 1; j < num_boxes; j++) {
            int next_idx = scores[j].index;
            if (suppressed[next_idx]) {
                continue;
            }

            // Suppress boxes of the same class (or general overlap) exceeding IoU threshold
            if (boxes_in[orig_idx].class_id == boxes_in[next_idx].class_id || iou_threshold < 0.3f) {
                float iou = compute_iou(&boxes_in[orig_idx], &boxes_in[next_idx]);
                if (iou >= iou_threshold) {
                    suppressed[next_idx] = 1;
                }
            }
        }
    }

    return count;
}
