"""
ASR 会话状态 management —— 每个 WebSocket 连接对应一个 ASRSession。
"""

from __future__ import annotations

import enum
import random
import string

from src.core.config import settings
from src.services.asr_service import build_hotword_context
from src.services.vad_service import StreamingVADSession


def _generate_sid() -> str:
    """生成会话 ID，格式 AST_XXXXXXXXXXXX。"""
    chars = string.ascii_uppercase + string.digits
    suffix = "".join(random.choices(chars, k=13))
    return f"AST_{suffix}"


class SessionState(enum.Enum):
    HANDSHAKING = "handshaking"
    STREAMING = "streaming"
    CLOSING = "closing"


class ASRSession:
    """
    单个 WebSocket 连接的会话上下文。

    维护段序号、时间偏移、热词等。
    每个 Session 持有独立的 StreamingVADSession 实例，连接创建时初始化，
    连接关闭时随 Session 对象销毁。
    """

    def __init__(self, trace_id: str, biz_id: str, app_id: str = "") -> None:
        self.sid: str = _generate_sid()
        self.trace_id: str = trace_id
        self.biz_id: str = biz_id
        self.app_id: str = app_id
        self.state: SessionState = SessionState.HANDSHAKING

        # 段序号（每次断句 +1）
        self.seg_id: int = 0

        # 热词上下文（默认从环境变量 HOTWORDS 读取，客户端可追加）
        self.hotword_context: str = build_hotword_context(settings.HOTWORDS)

        # 每连接独立的 VAD 实例
        self.vad: StreamingVADSession = StreamingVADSession()

    def next_seg_id(self) -> int:
        """获取当前段号并递增。"""
        current = self.seg_id
        self.seg_id += 1
        return current

    def set_streaming(self) -> None:
        self.state = SessionState.STREAMING

    def set_closing(self) -> None:
        self.state = SessionState.CLOSING
