"""
音频处理工具函数。
"""

import base64

import numpy as np


SAMPLE_RATE = 16000  # 16kHz
SAMPLE_WIDTH = 2  # 16bit = 2 bytes


def decode_base64_pcm(b64_data: str) -> np.ndarray:
    """
    将 Base64 编码的 PCM 16k/16bit 数据解码为 int16 numpy 数组。

    Args:
        b64_data: Base64 编码的原始 PCM 字节流

    Returns:
        int16 numpy 数组
    """
    raw_bytes = base64.b64decode(b64_data)
    return np.frombuffer(raw_bytes, dtype=np.int16).copy()


def int16_to_float32(audio: np.ndarray) -> np.ndarray:
    """int16 → float32（归一化到 [-1, 1]）。"""
    return audio.astype(np.float32) / 32768.0


def samples_to_ms(samples: int, sr: int = SAMPLE_RATE) -> int:
    """采样数 → 毫秒。"""
    return int(samples * 1000 / sr)


def samples_to_cs(samples: int, sr: int = SAMPLE_RATE) -> int:
    """采样数 → 厘秒（10ms 为 1 厘秒）。"""
    return int(samples * 100 / sr)
