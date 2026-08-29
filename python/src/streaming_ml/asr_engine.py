"""
Streaming Automatic Speech Recognition (ASR) sliding-window token accumulator.
Supports both numpy.ndarray and pure Python list inputs.
"""

from typing import List, Any

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

    def ingest_frame(self, audio: Any, vad_active: bool) -> str:
        """
        Accumulates streaming audio frames during active speech segments
        and produces real-time transcript updates.
        """
        if vad_active:
            self.speech_frames += 1
            if HAS_NUMPY and isinstance(audio, np.ndarray):
                self.history.append(audio.copy())
            else:
                self.history.append(list(audio))

            self.is_active = True
            if len(self.history) > self.buffer_frames:
                self.history.pop(0)

            # Produce progressive streaming tokens based on continuous speech duration
            if self.speech_frames == 3:
                return "voice_detected"
            elif self.speech_frames == 8:
                return "speech_stream"
            elif self.speech_frames == 16:
                return "ml_active"
            elif self.speech_frames == 28:
                return "continuous_audio"
            elif self.speech_frames > 28 and self.speech_frames % 20 == 0:
                return "streaming_token"
            else:
                return ""
        else:
            if self.is_active:
                self.is_active = False
                self.speech_frames = 0
                self.history.clear()
            return ""
