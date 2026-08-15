"""
Utilities for zero-copy memory validation and tensor preprocessing.
Supports numpy.ndarray when available, with pure-Python sequence fallback.
"""

from typing import Any, Sequence

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def validate_audio_tensor(audio: Any, expected_samples: int = 512) -> Any:
    """
    Validates that incoming audio memory is a contiguous float32 1D buffer.
    """
    if HAS_NUMPY:
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio, dtype=np.float32)

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        if not audio.flags["C_CONTIGUOUS"]:
            audio = np.ascontiguousarray(audio)

        if audio.ndim != 1:
            audio = audio.reshape(-1)

        if len(audio) != expected_samples:
            raise ValueError(f"Audio buffer length mismatch: expected {expected_samples}, got {len(audio)}")

        return audio
    else:
        if not isinstance(audio, (list, tuple, Sequence)):
            raise TypeError(f"Expected sequence, received {type(audio)}")
        if len(audio) != expected_samples:
            raise ValueError(f"Audio buffer length mismatch: expected {expected_samples}, got {len(audio)}")
        return [float(x) for x in audio]
