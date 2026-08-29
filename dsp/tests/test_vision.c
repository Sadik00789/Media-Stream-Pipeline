#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include "dsp_core.h"
#include "simd_vision.h"

void test_rgb24_to_planar(void) {
    printf("[TEST] Testing SIMD RGB24 to Planar float32 normalization...\n");
    const int width = 640;
    const int height = 480;
    const int total_pixels = width * height;

    uint8_t* rgb = (uint8_t*)malloc(total_pixels * 3);
    float* planar = (float*)malloc(total_pixels * 3 * sizeof(float));
    assert(rgb != NULL && planar != NULL);

    // Populate known pattern: pixel 0 = (255, 0, 128), pixel 1 = (0, 255, 64), etc.
    for (int i = 0; i < total_pixels; i++) {
        rgb[i * 3 + 0] = (uint8_t)((i * 7) % 256);
        rgb[i * 3 + 1] = (uint8_t)((i * 13) % 256);
        rgb[i * 3 + 2] = (uint8_t)((i * 19) % 256);
    }

    simd_rgb24_to_planar_f32(rgb, planar, width, height);

    float* r_plane = planar;
    float* g_plane = planar + total_pixels;
    float* b_plane = planar + (2 * total_pixels);

    for (int i = 0; i < total_pixels; i++) {
        float expected_r = (float)rgb[i * 3 + 0] / 255.0f;
        float expected_g = (float)rgb[i * 3 + 1] / 255.0f;
        float expected_b = (float)rgb[i * 3 + 2] / 255.0f;

        if (fabsf(r_plane[i] - expected_r) >= 1e-4f ||
            fabsf(g_plane[i] - expected_g) >= 1e-4f ||
            fabsf(b_plane[i] - expected_b) >= 1e-4f) {
            fprintf(stderr, "Mismatch at pixel %d\n", i);
            abort();
        }
    }

    free(rgb);
    free(planar);
    printf("  -> RGB24 to planar f32 conversion verified successfully.\n");
}

void test_fast_box_nms(void) {
    printf("[TEST] Testing fast bounding box Non-Maximum Suppression (NMS)...\n");

    // Create 4 test bounding boxes:
    // Box 0: High confidence face box [0.1, 0.1, 0.5, 0.5], conf = 0.95, class = 0
    // Box 1: Highly overlapping candidate [0.12, 0.11, 0.49, 0.51], conf = 0.85, class = 0 (Should be suppressed)
    // Box 2: Non-overlapping person box [0.6, 0.6, 0.9, 0.9], conf = 0.90, class = 0 (Should be kept)
    // Box 3: Low confidence candidate [0.62, 0.61, 0.88, 0.89], conf = 0.40, class = 0 (Should be suppressed by Box 2)

    DetectionBox boxes_in[4] = {
        { .ymin = 0.10f, .xmin = 0.10f, .ymax = 0.50f, .xmax = 0.50f, .confidence = 0.95f, .class_id = 0 },
        { .ymin = 0.12f, .xmin = 0.11f, .ymax = 0.49f, .xmax = 0.51f, .confidence = 0.85f, .class_id = 0 },
        { .ymin = 0.60f, .xmin = 0.60f, .ymax = 0.90f, .xmax = 0.90f, .confidence = 0.90f, .class_id = 0 },
        { .ymin = 0.62f, .xmin = 0.61f, .ymax = 0.88f, .xmax = 0.89f, .confidence = 0.40f, .class_id = 0 },
    };

    DetectionBox boxes_out[4];
    memset(boxes_out, 0, sizeof(boxes_out));

    uint32_t kept = simd_fast_box_nms(boxes_in, 4, 0.45f, boxes_out, 4);

    if (kept != 2 || fabsf(boxes_out[0].confidence - 0.95f) >= 1e-4f || fabsf(boxes_out[1].confidence - 0.90f) >= 1e-4f) {
        fprintf(stderr, "NMS test failed (kept = %u)\n", kept);
        abort();
    }

    printf("  -> NMS filtered 4 candidates down to 2 distinct boxes as expected.\n");
}

int main(void) {
    printf("[TEST] Running SIMD Vision Kernels verification suite...\n");
    test_rgb24_to_planar();
    test_fast_box_nms();
    printf("[PASS] All SIMD Vision tests passed successfully!\n");
    return 0;
}
