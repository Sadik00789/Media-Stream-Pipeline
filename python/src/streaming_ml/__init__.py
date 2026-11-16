"""
Streaming ML Engine package for real-time audio and vision inference.
"""

from .inference import StreamingInferenceEngine
from .asr_engine import StreamingAsrEngine
from .vision_engine import VisionEngine
from .tensor_utils import validate_audio_tensor

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "StreamingInferenceEngine",
    "StreamingAsrEngine",
    "VisionEngine",
    "validate_audio_tensor",
]
