"""
Unit tests for Python streaming ML engine.
"""

import math
from streaming_ml.inference import StreamingInferenceEngine
from streaming_ml.tensor_utils import validate_audio_tensor


def test_tensor_validation():
    # Valid input
    valid_buf = [0.0] * 512
    res = validate_audio_tensor(valid_buf)
    assert len(res) == 512

    # Invalid length
    invalid_buf = [0.0] * 256
    try:
        validate_audio_tensor(invalid_buf)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_inference_engine_silence():
    engine = StreamingInferenceEngine()
    silence = [0.0] * 512
    conf, text = engine.process_frame(silence)
    assert 0.0 <= conf <= 1.0
    assert conf < 0.2


def test_inference_engine_loud_tone():
    engine = StreamingInferenceEngine()
    tone = [0.5 * math.sin(2.0 * math.pi * 1000.0 * float(i) / 16000.0) for i in range(512)]
    conf, text = engine.process_frame(tone)
    assert 0.0 <= conf <= 1.0
    assert conf > 0.6


def test_nan_inf_sanitization():
    nan_buf = [float("nan")] * 256 + [float("inf")] * 128 + [float("-inf")] * 128
    cleaned = validate_audio_tensor(nan_buf)
    assert len(cleaned) == 512
    assert all(math.isfinite(float(x)) for x in cleaned)

    engine = StreamingInferenceEngine()
    conf, text = engine.process_frame(nan_buf)
    assert math.isfinite(conf)
    assert conf == 0.0
    assert text == ""


def test_zero_byte_onnx_fallback_and_process_pcm(tmp_path):
    empty_model = tmp_path / "empty_silero.onnx"
    empty_model.write_bytes(b"")

    engine = StreamingInferenceEngine(model_path=str(empty_model))
    assert engine.session is None

    tone = [0.5 * math.sin(2.0 * math.pi * 1000.0 * float(i) / 16000.0) for i in range(512)]
    conf = engine.process_pcm(tone, vad_active=True)
    assert 0.6 < conf <= 1.0


if __name__ == "__main__":
    print("[TEST] Running Python Streaming ML verification suite...")
    test_tensor_validation()
    print("  -> Tensor validation test passed")
    test_inference_engine_silence()
    print("  -> Silence inference test passed")
    test_inference_engine_loud_tone()
    print("  -> Loud tone speech inference test passed")
    print("[PASS] Python ML tests passed successfully!")
