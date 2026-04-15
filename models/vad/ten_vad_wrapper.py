"""
TEN VAD wrapper.
"""
import importlib.util
import os
import sys

import numpy as np


class TenVADWrapper:
    """TEN VAD model wrapper."""

    def __init__(
        self,
        hop_size=256,
        threshold=0.5,
        sample_rate=16000,
        min_silence_duration_ms=0.0,
    ):
        self.hop_size = hop_size
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.min_silence_duration_ms = float(min_silence_duration_ms)
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

    def get_speech_segments(self, audio, sr, min_silence_duration_ms=None):
        if audio.ndim > 1:
            audio = audio[:, 0]

        if sr != self.sample_rate:
            raise ValueError(
                f"TEN VAD only supports {self.sample_rate}Hz currently, got {sr}Hz"
            )

        audio_i16 = self._to_int16(audio)
        total_samples = len(audio_i16)
        if min_silence_duration_ms is None:
            min_silence_duration_ms = self.min_silence_duration_ms

        frame_ms = self.hop_size * 1000.0 / self.sample_rate
        min_silence_frames = 1
        if min_silence_duration_ms > 0:
            min_silence_frames = max(
                1, int(np.ceil(min_silence_duration_ms / frame_ms))
            )

        remainder = len(audio_i16) % self.hop_size
        if remainder != 0:
            pad_len = self.hop_size - remainder
            audio_i16 = np.pad(audio_i16, (0, pad_len), mode="constant")

        frame_count = len(audio_i16) // self.hop_size
        flags = np.zeros(frame_count, dtype=np.int32)

        for i in range(frame_count):
            frame = audio_i16[i * self.hop_size : (i + 1) * self.hop_size]
            _, out_flag = self._ten_vad.process(frame)
            flags[i] = out_flag

        return self._flags_to_segments(
            flags=flags,
            total_samples=total_samples,
            min_silence_frames=min_silence_frames,
        )

    def _flags_to_segments(self, flags, total_samples, min_silence_frames):
        segments = []
        in_speech = False
        start_frame = 0
        silence_frames = 0

        for i, flag in enumerate(flags):
            if flag == 1 and not in_speech:
                in_speech = True
                start_frame = i
                silence_frames = 0
                continue

            if flag == 1:
                silence_frames = 0
                continue

            if not in_speech:
                continue

            silence_frames += 1
            if silence_frames < min_silence_frames:
                continue

            end_frame = i - silence_frames + 1
            if end_frame > start_frame:
                segments.append(
                    (
                        start_frame * self.hop_size,
                        min(end_frame * self.hop_size, total_samples),
                    )
                )
            in_speech = False
            silence_frames = 0

        if in_speech:
            segments.append((start_frame * self.hop_size, total_samples))

        return segments

    @staticmethod
    def _to_int16(audio):
        if np.issubdtype(audio.dtype, np.integer):
            if audio.dtype == np.int16:
                return audio
            return audio.astype(np.int16)
        clipped = np.clip(audio, -1.0, 1.0)
        return (clipped * 32767.0).astype(np.int16)
