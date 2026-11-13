"""
Unit tests for Python Streaming ASR Engine.
"""

import math
import numpy as np
from streaming_ml.asr_engine import StreamingAsrEngine


def test_asr_engine_token_progression_and_vocabulary():
    engine = StreamingAsrEngine(sample_rate=16000)
    tone = [0.5 * math.sin(2.0 * math.pi * 1000.0 * float(i) / 16000.0) for i in range(512)]

    tokens = []
    for _ in range(16):
        conf, token = engine.process_frame(tone, vad_active=True)
        assert 0.65 <= conf <= 1.0
        if token:
            tokens.append(token)

    assert "voice_detected" in tokens
    assert any(t in engine.vocabulary for t in tokens)


def test_asr_engine_silence_decay_and_numpy_input():
    engine = StreamingAsrEngine()
    loud = np.full(512, 0.3, dtype=np.float32)
    silence = np.zeros(512, dtype=np.float32)

    for _ in range(4):
        engine.process_frame(loud, vad_active=True)
    assert engine.speech_frames == 4

    conf, token = engine.process_frame(silence, vad_active=False)
    assert conf == 0.0
    assert token == ""
    assert engine.speech_frames == 3

    # ingest_frame convenience wrapper
    tok = engine.ingest_frame(silence, vad_active=False)
    assert tok == ""

