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

        # Resolve model path dynamically across different working directories
        resolved_path = None
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            model_path,
            os.path.join(base_dir, "../../models/silero_vad.onnx"),
            os.path.join(base_dir, "../../../python/models/silero_vad.onnx"),
            os.path.join(base_dir, "../models/silero_vad.onnx"),
            os.path.join(os.getcwd(), "python/models/silero_vad.onnx"),
            os.path.join(os.getcwd(), "models/silero_vad.onnx"),
            os.path.join(os.getcwd(), "../../python/models/silero_vad.onnx"),
        ]

        for cand in candidates:
            if cand and os.path.exists(cand):
                resolved_path = os.path.abspath(cand)
                break

        if HAS_ONNX and HAS_NUMPY and resolved_path:
            try:
                opts = ort.SessionOptions()
                opts.inter_op_num_threads = 1
                opts.intra_op_num_threads = 1
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.session = ort.InferenceSession(
                    resolved_path,
                    sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
                self.model_path = resolved_path
            except Exception as e:
                print(f"[StreamingML] Warning: Failed to load ONNX model ({e}), using acoustic fallback.")
                self.session = None

    def process_frame(self, audio_array: Any, vad_active: Optional[bool] = None) -> Tuple[float, str]:
        """
        Processes a raw 512-sample audio frame.
        Args:
            audio_array: 512 PCM float32 samples (NumPy array or Python sequence).
            vad_active: Optional boolean indicating upstream hardware VAD state.
        Returns:
            (confidence_score: float, transcript_token: str)
        """
        audio = validate_audio_tensor(audio_array, expected_samples=512)

        # 1. ONNX Model Inference (if loaded successfully)
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

                is_speech = prob > 0.5 if vad_active is None else (vad_active or prob > 0.5)
                token = self.asr.ingest_frame(audio, is_speech)
                return prob, token
            except Exception:
                pass

        # 2. High-performance Acoustic Fallback Inference
        if HAS_NUMPY and isinstance(audio, np.ndarray):
            energy = float(np.mean(audio ** 2))
            zcr = float(np.mean(np.abs(np.diff(np.sign(audio + 1e-12)))) / 2.0)
        else:
            n = len(audio)
            energy = sum(x * x for x in audio) / float(n) if n > 0 else 0.0
            zcr_count = sum(1 for i in range(n - 1) if (audio[i] >= 0) != (audio[i + 1] >= 0))
            zcr = float(zcr_count) / float(n) if n > 0 else 0.0

        if energy < 1e-6:
            self.asr.ingest_frame(audio, False)
            return 0.0, ""

        rms = math.sqrt(energy)
        clamped_zcr = min(0.8, max(0.0, zcr))

        # Human voice & test tone signals have strong RMS and structured spectral energy
        raw_score = (rms * 14.0) * (1.0 - 0.4 * clamped_zcr)
        confidence = 1.0 / (1.0 + math.exp(-10.0 * (raw_score - 0.20)))
        confidence = max(0.0, min(1.0, float(confidence)))

        is_voice = (vad_active is True) or (confidence > 0.40) or (energy > 0.001)
        transcript = self.asr.ingest_frame(audio, is_voice)

        return confidence, transcript
