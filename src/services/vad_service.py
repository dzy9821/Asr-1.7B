"""
流式 VAD 服务 —— 基于 TenVADWrapper 做实时逐帧适配。

每个 WebSocket 连接持有一个 StreamingVADSession 实例。
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np

from src.core.logging import get_logger

logger = get_logger(__name__)

# ---- 动态阈值参数（与设计文档一致） ----
T_MAX = 2.0   # 语音极短时最长等待停顿（秒）
T_MIN = 0.3   # 语音极长时最短等待停顿（秒）
K = 0.17      # 停顿阈值递减斜率（秒/秒）
MIN_SPEECH_DURATION = 0.5   # 短音频抑制门限（秒）
MAX_SPEECH_DURATION = 15.0  # 长音频强制触发门限（秒）


def _load_ten_vad(hop_size: int = 640, threshold: float = 0.5):
    """加载底层 TenVad C 运行时实例。"""
    vad_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "models", "vad"
    )
    vad_dir = os.path.abspath(vad_dir)

    # 复用 TenVADWrapper 的加载逻辑
    sys.path.insert(0, vad_dir)
    from ten_vad_wrapper import TenVADWrapper  # noqa: E402

    wrapper = TenVADWrapper(hop_size=hop_size, threshold=threshold)
    return wrapper._ten_vad


class StreamingVADSession:
    """
    流式 VAD 会话。

    逐帧接收 PCM int16 音频，实时判定语音/静默状态，
    并在满足动态阈值条件时返回完整的语音片段。
    """

    def __init__(
        self,
        hop_size: int = 640,
        threshold: float = 0.5,
        sample_rate: int = 16000,
    ):
        self.hop_size = hop_size
        self.sample_rate = sample_rate
        self.frame_duration = hop_size / sample_rate  # 40ms

        self._vad = _load_ten_vad(hop_size, threshold)

        # 样本缓冲（不足一帧时暂存）
        self._sample_buffer = np.array([], dtype=np.int16)

        # 当前语音段的帧列表
        self._speech_frames: list[np.ndarray] = []
        self._in_speech = False
        self._speech_frame_count = 0  # 语音帧数
        self._silence_frame_count = 0  # 连续静默帧数

        # 全局采样计数（用于时间戳计算）
        self._total_samples: int = 0
        self._speech_start_sample: int = 0

    # ---- 公开接口 ----

    def feed_audio(self, pcm_int16: np.ndarray) -> list[dict]:
        """
        喂入 PCM int16 音频样本。

        Returns:
            触发的语音段列表，每项为
            {"audio": np.ndarray (int16), "start_sample": int, "end_sample": int}
        """
        self._sample_buffer = np.concatenate([self._sample_buffer, pcm_int16])
        segments: list[dict] = []

        while len(self._sample_buffer) >= self.hop_size:
            frame = self._sample_buffer[: self.hop_size]
            self._sample_buffer = self._sample_buffer[self.hop_size :]

            result = self._process_frame(frame)
            if result is not None:
                segments.append(result)

        return segments

    def flush(self) -> Optional[dict]:
        """
        强制刷出剩余语音段（客户端发送 status=2 时调用）。
        """
        if not self._speech_frames:
            return None

        speech_duration = self._speech_frame_count * self.frame_duration
        if speech_duration < MIN_SPEECH_DURATION:
            self._reset()
            return None

        return self._extract_and_reset()

    # ---- 内部逻辑 ----

    def _process_frame(self, frame: np.ndarray) -> Optional[dict]:
        _, flag = self._vad.process(frame)
        self._total_samples += self.hop_size

        if flag == 1:  # 语音
            if not self._in_speech:
                self._in_speech = True
                self._silence_frame_count = 0
                self._speech_frame_count = 0
                self._speech_start_sample = self._total_samples - self.hop_size
            self._speech_frame_count += 1
            self._silence_frame_count = 0
            self._speech_frames.append(frame)
        else:  # 静默
            if self._in_speech:
                self._speech_frames.append(frame)
                self._silence_frame_count += 1

                speech_dur = self._speech_frame_count * self.frame_duration
                pause_dur = self._silence_frame_count * self.frame_duration

                if _should_transcribe(speech_dur, pause_dur):
                    return self._extract_and_reset()

        # 强制触发：语音过长
        if self._in_speech:
            speech_dur = self._speech_frame_count * self.frame_duration
            if speech_dur > MAX_SPEECH_DURATION:
                return self._extract_and_reset()

        return None

    def _extract_and_reset(self) -> dict:
        audio = np.concatenate(self._speech_frames)
        start = self._speech_start_sample
        end = start + len(audio)
        self._reset()
        return {"audio": audio, "start_sample": start, "end_sample": end}

    def _reset(self) -> None:
        self._speech_frames = []
        self._in_speech = False
        self._speech_frame_count = 0
        self._silence_frame_count = 0


# ---- 动态阈值判定（独立函数，方便单元测试） ----


def _should_transcribe(speech_duration: float, pause_duration: float) -> bool:
    """
    判断是否应触发转写。

    规则：停顿阈值随语音时长线性递减。
    """
    if speech_duration < MIN_SPEECH_DURATION:
        return False
    if speech_duration > MAX_SPEECH_DURATION:
        return True

    threshold = max(T_MIN, T_MAX - K * speech_duration)
    return pause_duration >= threshold
