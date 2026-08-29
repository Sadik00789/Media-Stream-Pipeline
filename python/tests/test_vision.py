"""
Unit tests for Python streaming Vision ML engine.
"""

import numpy as np
from streaming_ml.vision_engine import VisionEngine


def test_vision_engine_init():
    engine = VisionEngine()
    assert engine.width == 640
    assert engine.height == 480
    assert engine.channels == 3


def test_vision_engine_blank_frame():
    engine = VisionEngine()
    dark_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    boxes = engine.process_frame(dark_frame)
    assert isinstance(boxes, list)
    assert len(boxes) == 0


def test_vision_engine_synthetic_face():
    engine = VisionEngine()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Synthetic skin-tone rectangle in center (RGB ~ 220, 160, 120)
    frame[150:330, 200:440, 0] = 220
    frame[150:330, 200:440, 1] = 160
    frame[150:330, 200:440, 2] = 120

    boxes = engine.process_frame(frame)
    assert len(boxes) >= 1

    box = boxes[0]
    assert len(box) == 6
    ymin, xmin, ymax, xmax, conf, class_id = box

    assert 0.0 <= ymin <= ymax <= 1.0
    assert 0.0 <= xmin <= xmax <= 1.0
    assert conf > 0.60
    assert class_id == 0  # Face class


def test_vision_engine_flat_array_input():
    engine = VisionEngine()
    flat_frame = np.zeros(640 * 480 * 3, dtype=np.uint8)
    boxes = engine.process_frame(flat_frame)
    assert isinstance(boxes, list)


if __name__ == "__main__":
    print("[TEST] Running Python Streaming Vision ML test suite...")
    test_vision_engine_init()
    print("  -> Vision engine initialization passed")
    test_vision_engine_blank_frame()
    print("  -> Blank frame test passed")
    test_vision_engine_synthetic_face()
    print("  -> Synthetic face detection test passed")
    test_vision_engine_flat_array_input()
    print("  -> Flat array input test passed")
    print("[PASS] All Python Vision ML tests passed successfully!")
