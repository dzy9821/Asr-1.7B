"""
FSMN VAD wrapper via ModelScope/FunASR pipeline.
"""
import os

import numpy as np


class FSMNVADWrapper:
    """FSMN VAD model wrapper."""

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__),
                "speech_fsmn_vad_zh-cn-16k-common-pytorch",
            )
        self.model_path = os.path.abspath(model_path)
        self._pipeline = self._load_pipeline()

    def _load_pipeline(self):
        try:
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks
        except Exception as exc:
            raise ImportError("ModelScope is required for FSMN VAD wrapper") from exc

        try:
            return pipeline(
                task=Tasks.voice_activity_detection,
                model=self.model_path,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize FSMN VAD pipeline. "
                "Please check runtime dependencies (e.g., torchaudio)."
            ) from exc

    def get_speech_segments(self, audio, sr):
        output = self._pipeline(input=audio, fs=sr)
        pairs = []
        self._collect_pairs(output, pairs)
        if not pairs:
            return []

        segments = []
        max_end = max(int(p[1]) for p in pairs)
        audio_len = len(audio)
        looks_like_ms = max_end > audio_len and max_end <= int(audio_len / sr * 1000.0) + 1000

        for start, end in pairs:
            start_i = int(start)
            end_i = int(end)
            if looks_like_ms:
                start_i = int(start_i * sr / 1000.0)
                end_i = int(end_i * sr / 1000.0)
            start_i = max(0, min(start_i, audio_len))
            end_i = max(0, min(end_i, audio_len))
            if end_i > start_i:
                segments.append((start_i, end_i))
        return segments

    def _collect_pairs(self, obj, out):
        if isinstance(obj, (list, tuple)):
            if len(obj) == 2 and self._is_number(obj[0]) and self._is_number(obj[1]):
                out.append((obj[0], obj[1]))
                return
            for item in obj:
                self._collect_pairs(item, out)
            return

        if isinstance(obj, dict):
            for value in obj.values():
                self._collect_pairs(value, out)

    @staticmethod
    def _is_number(v):
        return isinstance(v, (int, float, np.integer, np.floating))

