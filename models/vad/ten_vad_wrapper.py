"""
TEN VAD wrapper.
"""
import importlib.util
import os
import sys

import numpy as np


class TenVADWrapper:
    """TEN VAD model wrapper."""

    def __init__(self, hop_size=256, threshold=0.5, sample_rate=16000):
        self.hop_size = hop_size
        self.threshold = threshold
        self.sample_rate = sample_rate
        self._ten_vad = self._load_model()

    def _load_model(self):
        include_dir = os.path.join(os.path.dirname(__file__), "ten-vad/include")
        module_path = os.path.join(include_dir, "ten_vad.py")
        if not os.path.exists(module_path):
            raise FileNotFoundError(f"ten_vad.py not found: {module_path}")

        if include_dir not in sys.path:
            sys.path.insert(0, include_dir)

        spec = importlib.util.spec_from_file_location("ten_vad_local", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load module spec from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        ten_vad_cls = getattr(module, "TenVad")
        try:
            return ten_vad_cls(self.hop_size, self.threshold)
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize TEN VAD runtime. "
                "Make sure required native libs are available."
            ) from exc

    def get_speech_segments(self, audio, sr):
        if audio.ndim > 1:
            audio = audio[:, 0]

        if sr != self.sample_rate:
            raise ValueError(
                f"TEN VAD only supports {self.sample_rate}Hz currently, got {sr}Hz"
            )

        audio_i16 = self._to_int16(audio)
        frame_count = len(audio_i16) // self.hop_size
        flags = np.zeros(frame_count, dtype=np.int32)

        for i in range(frame_count):
            frame = audio_i16[i * self.hop_size : (i + 1) * self.hop_size]
            _, out_flag = self._ten_vad.process(frame)
            flags[i] = out_flag

        return self._flags_to_segments(flags)

    def _flags_to_segments(self, flags):
        segments = []
        in_speech = False
        start = 0

        for i, flag in enumerate(flags):
            frame_start = i * self.hop_size
            if flag == 1 and not in_speech:
                in_speech = True
                start = frame_start
            elif flag == 0 and in_speech:
                in_speech = False
                segments.append((start, frame_start))

        if in_speech:
            segments.append((start, len(flags) * self.hop_size))

        return segments

    @staticmethod
    def _to_int16(audio):
        if np.issubdtype(audio.dtype, np.integer):
            if audio.dtype == np.int16:
                return audio
            return audio.astype(np.int16)
        clipped = np.clip(audio, -1.0, 1.0)
        return (clipped * 32767.0).astype(np.int16)

