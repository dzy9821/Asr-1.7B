"""
VAD模型包装器 - Silero VAD
"""
import os
import numpy as np
import onnxruntime as rt


class SileroVAD:
    """Silero VAD模型包装"""

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__), "silero-vad-onnx/silero_vad.onnx"
            )
        self.model_path = model_path
        self.session = rt.InferenceSession(model_path)
        self.sample_rate = 16000

    def get_speech_segments(self, audio, sr, threshold=0.5):
        """
        获取语音片段的起止位置

        Args:
            audio: 音频数组
            sr: 采样率
            threshold: VAD阈值

        Returns:
            segments: [(start_sample, end_sample), ...] 列表
        """
        # 重采样到16kHz
        if sr != self.sample_rate:
            audio = self._resample(audio, sr, self.sample_rate)

        # 转换为float32
        audio = audio.astype(np.float32)

        # 分帧处理
        frame_size = int(self.sample_rate * 0.032)  # 32ms
        segments = []
        in_speech = False
        segment_start = 0

        # 初始化隐状态
        h = np.zeros((2, 1, 64), dtype=np.float32)
        c = np.zeros((2, 1, 64), dtype=np.float32)

        for i in range(0, len(audio) - frame_size, frame_size):
            frame = audio[i : i + frame_size]
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

            if prob > threshold and not in_speech:
                segment_start = i
                in_speech = True
            elif prob <= threshold and in_speech:
                segments.append((segment_start, i))
                in_speech = False

        if in_speech:
            segments.append((segment_start, len(audio)))

        return segments

    def _resample(self, audio, sr_orig, sr_target):
        """简单的重采样"""
        import librosa

        return librosa.resample(audio, orig_sr=sr_orig, target_sr=sr_target)
