/**
 * @file simd_vision.h
 * @brief High-performance C11 SIMD Vision Kernels for real-time video tensor processing.
 */

#ifndef SIMD_VISION_H
#define SIMD_VISION_H

#include <stdint.h>
#include <stddef.h>
#include "ipc_protocol.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Converts interleaved 8-bit RGB (HWC) into normalized float32 planar (CHW) tensors.
 * Output tensor values are normalized to the range [0.0, 1.0].
 * 
 * @param rgb_in Interleaved RGB24 input buffer of size width * height * 3.
 * @param planar_out Planar float32 output buffer of size 3 * width * height [R plane, G plane, B plane].
 * @param width Image width in pixels (e.g. 640).
 * @param height Image height in pixels (e.g. 480).
 */
void simd_rgb24_to_planar_f32(const uint8_t* rgb_in, float* planar_out, int width, int height);

/**
 * @brief Fast bounding box Non-Maximum Suppression (NMS) using IoU filtering.
 * 
 * @param boxes_in Array of candidate DetectionBox instances.
 * @param num_boxes Number of input candidate boxes.
 * @param iou_threshold IoU threshold for suppressing overlapping boxes (e.g. 0.45).
 * @param boxes_out Output buffer for selected non-overlapping detection boxes.
 * @param max_out Maximum number of boxes to write to boxes_out.
 * @return uint32_t Number of selected detection boxes.
 */
uint32_t simd_fast_box_nms(
    const DetectionBox* boxes_in,
    uint32_t num_boxes,
    float iou_threshold,
    DetectionBox* boxes_out,
    uint32_t max_out
);

#ifdef __cplusplus
}
#endif

#endif /* SIMD_VISION_H */
