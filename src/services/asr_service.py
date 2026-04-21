"""
异步 ASR 调用服务 —— 通过 httpx 调用 vLLM OpenAI 兼容接口。
"""

from __future__ import annotations

import base64
import io
import re
from typing import Optional

import httpx
import numpy as np

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


class ASRService:
    """异步 ASR 推理服务，封装对 vLLM 的 HTTP 调用。"""

    def __init__(self) -> None:
        self._client: Optional[httpx.AsyncClient] = None

    async def startup(self) -> None:
        """初始化 HTTP 客户端（应用启动时调用）。"""
        self._client = httpx.AsyncClient(timeout=60.0)
        logger.info(
            "ASR service started, vLLM endpoint: %s, model: %s",
            settings.VLLM_API_BASE,
            settings.VLLM_MODEL_NAME,
        )

    async def shutdown(self) -> None:
        """关闭 HTTP 客户端。"""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def recognize(
        self,
        audio_int16: np.ndarray,
        sr: int = 16000,
        context: str = "",
    ) -> str:
        """
        将音频片段发送至 vLLM 进行 ASR 推理。

        Args:
            audio_int16: int16 PCM 音频
            sr: 采样率
            context: 热词/系统提示词

        Returns:
            清洗后的识别文本
        """
        assert self._client is not None, "ASRService not started"

        data_url = _encode_audio_to_data_url(audio_int16, sr)

        messages: list[dict] = []
        if context:
            messages.append({"role": "system", "content": context})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": data_url}},
                ],
            }
        )

        url = f"{settings.VLLM_API_BASE}/chat/completions"
        payload = {
            "model": settings.VLLM_MODEL_NAME,
            "messages": messages,
        }
        headers = {}
        if settings.VLLM_API_KEY and settings.VLLM_API_KEY != "EMPTY":
            headers["Authorization"] = f"Bearer {settings.VLLM_API_KEY}"

        response = await self._client.post(url, json=payload, headers=headers)
        response.raise_for_status()

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return _clean_asr_output(content if isinstance(content, str) else str(content))

    async def is_available(self) -> bool:
        """检查 vLLM 服务是否可达。"""
        if self._client is None:
            return False
        try:
            resp = await self._client.get(f"{settings.VLLM_API_BASE}/models", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False


# ---- 工具函数（复用 pipeline_common.py 的逻辑） ----


def _encode_audio_to_data_url(audio_int16: np.ndarray, sr: int) -> str:
    """将 int16 音频编码为 data:audio/wav;base64,... 格式。"""
    import soundfile as sf

    audio_f32 = audio_int16.astype(np.float32) / 32768.0
    buf = io.BytesIO()
    sf.write(buf, audio_f32, sr, format="WAV")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:audio/wav;base64,{b64}"


def _clean_asr_output(text: str) -> str:
    """清洗 ASR 模型输出，移除 <asr_text> 等标记。"""
    text = text.strip()
    if "<asr_text>" not in text:
        return text
    parts = re.split(r"(?:language\s+[^\s<]+)?<asr_text>", text, flags=re.IGNORECASE)
    return "".join(part.strip() for part in parts if part.strip()).strip()


def build_hotword_context(hotwords: Optional[str]) -> str:
    """将客户端传入的热词构建为系统提示词。"""
    if not hotwords or not hotwords.strip():
        return ""
    words = [w.strip() for w in re.split(r"[,\n;，；、]+", hotwords) if w.strip()]
    if not words:
        return ""
    return f"热词：{'、'.join(words)}"
