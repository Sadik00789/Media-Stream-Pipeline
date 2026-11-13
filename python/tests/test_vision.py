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


def test_vision_engine_zero_byte_onnx_fallback(tmp_path):
    empty_model = tmp_path / "empty_vision.onnx"
    empty_model.write_bytes(b"")
    engine = VisionEngine(model_path=str(empty_model))
    assert engine.session is None


def test_vision_engine_salient_contrast_object():
    engine = VisionEngine()
    frame = np.full((480, 640, 3), 100, dtype=np.uint8)
    # High-contrast grayscale blocks in center (not skin-tone, visible across stride-16)
    for y in range(128, 352, 32):
        for x in range(160, 480, 32):
            frame[y : y + 16, x : x + 16, :] = 250
            frame[y + 16 : y + 32, x + 16 : x + 32, :] = 20

    boxes = engine.process_frame(frame)
    assert len(boxes) == 1
    assert boxes[0][5] == 1.0  # Salient Object class
    assert boxes[0][4] > 0.5


def test_vision_engine_invalid_shape_graceful():
    engine = VisionEngine()
    assert engine.process_frame(np.zeros((10, 10), dtype=np.uint8)) == []
    assert engine.process_frame(np.zeros((5, 5, 3), dtype=np.uint8)) == []


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
