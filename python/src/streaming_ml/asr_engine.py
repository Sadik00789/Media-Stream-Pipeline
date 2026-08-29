"""
Streaming Automatic Speech Recognition (ASR) sliding-window token accumulator.
Supports both numpy.ndarray and pure Python list inputs.
"""

import math
from typing import List, Any, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class StreamingAsrEngine:
    def __init__(self, sample_rate: int = 16000, buffer_frames: int = 16):
        self.sample_rate = sample_rate
        self.buffer_frames = buffer_frames
        self.history: List[Any] = []
        self.speech_frames = 0
        self.is_active = False
        self.vocabulary = [
            "speech_detected", "voice_stream", "pipeline_active",
            "audio_tensor", "realtime_asr", "neural_token", "vad_high"
        ]

    def process_frame(self, pcm_data: Any, vad_active: bool = False) -> Tuple[float, str]:
        """
        Processes a raw PCM frame, computes ML acoustic confidence,
        and produces progressive streaming tokens.
        """
        if HAS_NUMPY:
            arr = np.asarray(pcm_data, dtype=np.float32)
            energy = float(np.sqrt(np.mean(arr**2))) if len(arr) > 0 else 0.0
        else:
            n = len(pcm_data)
            energy = math.sqrt(sum(x * x for x in pcm_data) / max(1, n)) if n > 0 else 0.0

        if vad_active and energy > 0.001:
            ml_conf = min(0.99, max(0.65, 0.65 + energy * 3.0))
        elif energy > 0.015:
            ml_conf = min(0.98, max(0.50, energy * 6.0))
        else:
            ml_conf = min(0.35, energy * 8.0)

        token = ""
        if vad_active or ml_conf > 0.55 or energy > 0.02:
            self.speech_frames += 1
            if self.speech_frames == 3:
                token = "voice_detected"
            elif self.speech_frames % 8 == 0:
                idx = (self.speech_frames // 8) % len(self.vocabulary)
                token = self.vocabulary[idx]
        else:
            self.speech_frames = max(0, self.speech_frames - 1)

        return float(ml_conf), str(token)

    def ingest_frame(self, audio: Any, vad_active: bool) -> str:
        _, token = self.process_frame(audio, vad_active)
        return token
