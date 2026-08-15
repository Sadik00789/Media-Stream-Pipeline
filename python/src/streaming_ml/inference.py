"""
Streaming ML Inference Engine for sub-millisecond voice activity and acoustic probability scoring.
Supports ONNX Runtime, NumPy, and pure-Python execution paths.
"""

import math
import os
from typing import Tuple, Optional, Any
from .tensor_utils import validate_audio_tensor
from .asr_engine import StreamingAsrEngine

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


class StreamingInferenceEngine:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.session = None
        self._sr = 16000
        self.asr = StreamingAsrEngine(sample_rate=16000)

        if HAS_NUMPY:
            self._state = np.zeros((2, 1, 128), dtype=np.float32)
        else:
            self._state = None

        if HAS_ONNX and HAS_NUMPY and model_path and os.path.exists(model_path):
            try:
                opts = ort.SessionOptions()
                opts.inter_op_num_threads = 1
                opts.intra_op_num_threads = 1
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.session = ort.InferenceSession(
                    model_path,
                    sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
            except Exception as e:
                print(f"[StreamingML] Warning: Failed to load ONNX model ({e}), using acoustic fallback.")
                self.session = None

    def process_frame(self, audio_array: Any) -> Tuple[float, str]:
        """
        Processes a raw 512-sample audio frame.
        Returns:
            (confidence_score: float, transcript_token: str)
        """
        audio = validate_audio_tensor(audio_array, expected_samples=512)

        # 1. ONNX Model Inference (if active)
        if self.session is not None and HAS_NUMPY:
            try:
                inputs = {
                    "input": audio.reshape(1, -1),
                    "state": self._state,
                    "sr": np.array(self._sr, dtype=np.int64),
                }
                outs = self.session.run(None, inputs)
                prob = float(outs[0][0][0])
                if len(outs) > 1:
                    self._state = outs[1]
                token = self.asr.ingest_frame(audio, prob > 0.5)
                return prob, token
            except Exception:
                pass

        # 2. High-performance Acoustic Fallback Inference
        if HAS_NUMPY and isinstance(audio, np.ndarray):
            energy = float(np.mean(audio ** 2))
            zcr = float(np.mean(np.abs(np.diff(np.sign(audio + 1e-12)))) / 2.0)
        else:
            n = len(audio)
            energy = sum(x * x for x in audio) / float(n)
            zcr_count = sum(1 for i in range(n - 1) if (audio[i] >= 0) != (audio[i + 1] >= 0))
            zcr = float(zcr_count) / float(n)

        if energy < 1e-5:
            self.asr.ingest_frame(audio, False)
            return 0.0, ""

        # Human speech typically has moderate ZCR and high spectral energy concentration
        clamped_zcr = min(0.8, max(0.0, zcr))
        raw_score = (math.sqrt(energy) * 8.0) * (1.0 - clamped_zcr)
        confidence = 1.0 / (1.0 + math.exp(-12.0 * (raw_score - 0.35)))
        confidence = max(0.0, min(1.0, float(confidence)))

        is_voice = confidence > 0.60
        transcript = self.asr.ingest_frame(audio, is_voice)
        if not transcript and is_voice:
            transcript = "voice_active"

        return confidence, transcript
