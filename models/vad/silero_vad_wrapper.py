"""
VAD模型包装器 - Silero VAD
"""
import os
import numpy as np
import onnxruntime as rt


class SileroVAD:
    """Silero VAD模型包装"""

    def __init__(
        self,
        model_path=None,
        threshold=0.5,
        min_silence_duration_ms=0.0,
    ):
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__), "silero-vad-onnx/silero_vad.onnx"
            )
        self.model_path = model_path
        self.session = rt.InferenceSession(model_path)
        self.sample_rate = 16000
        self.threshold = float(threshold)
        self.min_silence_duration_ms = float(min_silence_duration_ms)

    def get_speech_segments(self, audio, sr, threshold=None, min_silence_duration_ms=None):
        """
        获取语音片段的起止位置

        Args:
            audio: 音频数组
            sr: 采样率
            threshold: VAD阈值
            min_silence_duration_ms: 连续静音多长才切分

        Returns:
            segments: [(start_sample, end_sample), ...] 列表
        """
        if threshold is None:
            threshold = self.threshold
        if min_silence_duration_ms is None:
            min_silence_duration_ms = self.min_silence_duration_ms

        # 重采样到16kHz
        if sr != self.sample_rate:
            audio = self._resample(audio, sr, self.sample_rate)

        # 转换为float32
        audio = audio.astype(np.float32)

        # 分帧处理
        frame_size = int(self.sample_rate * 0.032)  # 32ms
        frame_ms = frame_size * 1000.0 / self.sample_rate
        min_silence_frames = 1
        if min_silence_duration_ms > 0:
            min_silence_frames = max(
                1, int(np.ceil(min_silence_duration_ms / frame_ms))
            )

        # 初始化隐状态
        h = np.zeros((2, 1, 64), dtype=np.float32)
        c = np.zeros((2, 1, 64), dtype=np.float32)
        flags = []

        for i in range(0, len(audio), frame_size):
            frame = audio[i : i + frame_size]
            if len(frame) < frame_size:
                frame = np.pad(frame, (0, frame_size - len(frame)), mode="constant")
            frame = frame.reshape(1, -1)  # 添加batch维度

            # 运行VAD推理
            ort_inputs = {
                "input": frame,
                "sr": np.array([self.sample_rate], dtype=np.int64),
                "h": h,
                "c": c,
            }
            ort_outs = self.session.run(None, ort_inputs)
            prob = ort_outs[0][0][0]  # 语音概率
            h = ort_outs[1]
            c = ort_outs[2]
            flags.append(1 if prob > threshold else 0)

        return self._flags_to_segments(
            flags=flags,
            frame_size=frame_size,
            total_samples=len(audio),
            min_silence_frames=min_silence_frames,
        )

    @staticmethod
    def _flags_to_segments(flags, frame_size, total_samples, min_silence_frames):
        segments = []
        in_speech = False
        start_frame = 0
        silence_frames = 0

        for i, flag in enumerate(flags):
            if flag == 1:
                if not in_speech:
                    in_speech = True
                    start_frame = i
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
                        start_frame * frame_size,
                        min(end_frame * frame_size, total_samples),
                    )
                )
            in_speech = False
            silence_frames = 0

        if in_speech:
            segments.append((start_frame * frame_size, total_samples))

        return segments

    def _resample(self, audio, sr_orig, sr_target):
        """简单的重采样"""
        import librosa

        return librosa.resample(audio, orig_sr=sr_orig, target_sr=sr_target)
