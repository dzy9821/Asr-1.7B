"""
流式 VAD 服务 —— 基于 Silero VAD 的全局单实例动态批处理架构。

架构：
  - 全局共享一个 Silero JIT 模型权重（SileroVADBatchProcessor 单例）
  - 每个 WebSocket 连接的时序状态（context + hidden state）在服务端外部管理
  - 多个连接的帧通过 asyncio Queue 收集，凑批后统一推理
  - StreamingVADSession 保持 feed_audio / flush 接口，内部改为 async
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Optional

import numpy as np
import torch

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)

# ---- 动态阈值参数（从环境变量读取，可随时调整） ----
T_MAX = settings.VAD_PAUSE_MAX                # 累积语音 0s 时所需停顿（秒）
T_MIN = settings.VAD_PAUSE_MIN                # 累积语音 >= DYNAMIC_RANGE_END 时所需停顿（秒）
DYNAMIC_RANGE_END = settings.VAD_DYNAMIC_RANGE_END  # 动态线性区间终点（秒）
K = (T_MAX - T_MIN) / DYNAMIC_RANGE_END if DYNAMIC_RANGE_END > 0 else 0.0  # 停顿阈值递减斜率
MIN_SPEECH_DURATION = settings.VAD_MIN_SPEECH  # 短音频抑制门限（秒），不足则不转发
MAX_SPEECH_DURATION = settings.VAD_MAX_SPEECH  # 长音频强制触发门限（秒），立即转发

# ---- Silero VAD 常量 ----
SILERO_WINDOW_SIZE = 512   # 16kHz 下每帧 512 samples = 32ms
SILERO_CONTEXT_SIZE = 64   # 上下文窗口大小
SILERO_STATE_DIM = 128     # RNN 隐藏状态维度
SAMPLE_RATE = 16000


# ============================================================
# 全局批处理器
# ============================================================


class SileroVADBatchProcessor:
    """
    全局 Silero VAD 批处理器（单例）。

    - 加载单个 Silero JIT 模型
    - 管理所有活跃 session 的时序状态（context + hidden state）
    - 后台 asyncio Task 收集帧请求，凑批后统一推理，再分发结果
    """

    def __init__(
        self,
        max_batch_size: int = 256,
        max_wait_ms: float = 10.0,
    ):
        self._max_batch_size = max_batch_size
        self._max_wait_ms = max_wait_ms

        self._raw_model: torch.jit.ScriptModule | None = None
        self._states: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        # asyncio 运行时资源（在 start() 中创建）
        self._queue: asyncio.Queue | None = None
        self._batch_task: asyncio.Task | None = None
        self._running = False

    # ---- 生命周期 ----

    def load_model(self) -> None:
        """Eager load Silero VAD JIT 模型（同步，在 lifespan 启动阶段调用）。"""
        model_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "models", "vad", "silero-vad"
        )
        model_dir = os.path.abspath(model_dir)

        model, _ = torch.hub.load(
            repo_or_dir=model_dir, model="silero_vad", source="local"
        )
        torch.set_grad_enabled(False)
        self._raw_model = model._model  # 底层 JIT RNN，支持 batch forward
        logger.info("Silero VAD model loaded from %s", model_dir)

    async def start(self) -> None:
        """启动批处理后台循环（在 lifespan 中调用）。"""
        if self._raw_model is None:
            self.load_model()
        self._queue = asyncio.Queue()
        self._running = True
        self._batch_task = asyncio.create_task(self._batch_loop())
        logger.info(
            "Silero VAD batch processor started (max_batch=%d, max_wait=%.1fms)",
            self._max_batch_size,
            self._max_wait_ms,
        )

    async def stop(self) -> None:
        """停止批处理后台循环（在 lifespan 中调用）。"""
        self._running = False
        if self._batch_task:
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass
        logger.info("Silero VAD batch processor stopped")

    # ---- Session 注册 ----

    def register_session(self, sid: str) -> None:
        """为新连接注册独立的时序状态。"""
        self._states[sid] = (
            torch.zeros(1, SILERO_CONTEXT_SIZE),
            torch.zeros(2, 1, SILERO_STATE_DIM),
        )

    def unregister_session(self, sid: str) -> None:
        """移除连接的时序状态。"""
        self._states.pop(sid, None)

    # ---- 帧提交（异步） ----

    async def process_frame(self, sid: str, frame_tensor: torch.Tensor) -> float:
        """
        提交一帧音频至批处理队列，等待推理结果。

        Args:
            sid: 会话 ID
            frame_tensor: [1, 512] float32 tensor

        Returns:
            语音概率（0~1）
        """
        future = asyncio.get_event_loop().create_future()
        await self._queue.put((sid, frame_tensor, future))
        return await future

    # ---- 后台批处理循环 ----

    async def _batch_loop(self) -> None:
        """后台协程：收集帧 → 凑批 → 推理 → 分发结果。"""
        while self._running:
            batch: list[tuple[str, torch.Tensor, asyncio.Future]] = []

            # 等待第一个请求
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                batch.append(item)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            # 在时间窗口内收集更多请求
            deadline = time.monotonic() + self._max_wait_ms / 1000
            while len(batch) < self._max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self._queue.get_nowait()
                    batch.append(item)
                except asyncio.QueueEmpty:
                    # 短暂让出控制权，让其他协程有机会入队
                    await asyncio.sleep(0.0005)

            # 执行批量推理
            await self._execute_batch(batch)

    async def _execute_batch(
        self, batch: list[tuple[str, torch.Tensor, asyncio.Future]]
    ) -> None:
        """执行一次批量推理并将结果分发给各 future。"""
        if not batch:
            return

        valid_sids: list[str] = []
        valid_frames: list[torch.Tensor] = []
        valid_futures: list[asyncio.Future] = []

        for sid, frame, future in batch:
            if sid not in self._states:
                future.set_exception(RuntimeError(f"Session {sid} not registered"))
                continue
            valid_sids.append(sid)
            valid_frames.append(frame)
            valid_futures.append(future)

        if not valid_sids:
            return

        try:
            n = len(valid_sids)

            # 聚合各 session 的 context 和 state
            contexts = torch.cat(
                [self._states[sid][0] for sid in valid_sids], dim=0
            )  # [n, 64]
            states = torch.cat(
                [self._states[sid][1] for sid in valid_sids], dim=1
            )  # [2, n, 128]

            # 构建输入：[context | audio_chunk]
            audio_batch = torch.cat(valid_frames, dim=0)  # [n, 512]
            x = torch.cat([contexts, audio_batch], dim=1)  # [n, 576]

            # 批量推理（移至后台线程执行，彻底避免霸占主线程阻塞事件循环）
            out, new_states = await asyncio.to_thread(self._raw_model, x, states)

            # 更新 session 状态并分发结果
            for i, (sid, future) in enumerate(zip(valid_sids, valid_futures)):
                if sid not in self._states:
                    continue
                self._states[sid] = (
                    x[i : i + 1, -SILERO_CONTEXT_SIZE :].clone(),
                    new_states[:, i : i + 1, :].clone(),
                )
                future.set_result(out[i].item())

        except Exception as exc:
            logger.exception("Batch inference failed: %s", exc)
            for future in valid_futures:
                if not future.done():
                    future.set_exception(exc)


# 全局单例
vad_processor = SileroVADBatchProcessor()


# ============================================================
# 流式 VAD 会话（每连接一个）
# ============================================================


class StreamingVADSession:
    """
    流式 VAD 会话。

    逐帧接收 PCM int16 音频，通过全局批处理器获取语音概率，
    并在满足动态阈值条件时返回完整的语音片段。
    """

    def __init__(
        self,
        sid: str,
        threshold: float = 0.5,
        sample_rate: int = SAMPLE_RATE,
    ):
        self._sid = sid
        self._threshold = threshold
        self.sample_rate = sample_rate
        self.hop_size = SILERO_WINDOW_SIZE  # 512 samples = 32ms
        self.frame_duration = self.hop_size / sample_rate

        # 注册至全局批处理器
        vad_processor.register_session(sid)

        # 样本缓冲（不足一帧时暂存，chunk list 避免 np.concatenate 全量拷贝）
        self._chunks: list[np.ndarray] = []
        self._chunk_total: int = 0

        # 当前语音段的帧列表
        self._speech_frames: list[np.ndarray] = []
        self._in_speech = False
        self._speech_frame_count = 0  # 语音帧数
        self._silence_frame_count = 0  # 连续静默帧数

        # 全局采样计数（用于时间戳计算）
        self._total_samples: int = 0
        self._speech_start_sample: int = 0

    # ---- 公开接口 ----

    async def feed_audio(self, pcm_int16: np.ndarray) -> list[dict]:
        """
        喂入 PCM int16 音频样本。

        Returns:
            触发的语音段列表，每项为
            {"audio": np.ndarray (int16), "start_sample": int, "end_sample": int}
        """
        self._chunks.append(pcm_int16)
        self._chunk_total += len(pcm_int16)
        segments: list[dict] = []

        while self._chunk_total >= self.hop_size:
            buffer = np.concatenate(self._chunks)
            frame = buffer[: self.hop_size]
            remainder = buffer[self.hop_size :]
            self._chunks = [remainder] if len(remainder) > 0 else []
            self._chunk_total = len(remainder)

            result = await self._process_frame(frame)
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

    def close(self) -> None:
        """从全局批处理器注销（连接关闭时调用）。"""
        vad_processor.unregister_session(self._sid)

    # ---- 内部逻辑 ----

    async def _process_frame(self, frame: np.ndarray) -> Optional[dict]:
        # int16 → float32 归一化 → tensor [1, 512]
        frame_f32 = frame.astype(np.float32) / 32768.0
        frame_tensor = torch.from_numpy(frame_f32).unsqueeze(0)

        # 通过批处理器获取语音概率
        prob = await vad_processor.process_frame(self._sid, frame_tensor)
        self._total_samples += self.hop_size

        flag = 1 if prob >= self._threshold else 0

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

    动态转写触发规则（阈值来自 config.py，可通过环境变量覆盖）：
      - speech < MIN_SPEECH  → 不触发（短音频抑制）
      - speech >= MAX_SPEECH → 立即触发（强制上限）
      - 0s ~ DYNAMIC_RANGE_END → 停顿阈值从 T_MAX 线性递减至 T_MIN
      - DYNAMIC_RANGE_END ~ MAX_SPEECH → 停顿阈值固定为 T_MIN
    """
    if speech_duration < MIN_SPEECH_DURATION:
        return False
    if speech_duration >= MAX_SPEECH_DURATION:
        return True

    if speech_duration >= DYNAMIC_RANGE_END:
        # 20~30s 区间：使用最小停顿阈值
        threshold = T_MIN
    else:
        # 0~20s 区间：线性递减
        threshold = T_MAX - K * speech_duration

    return pause_duration >= threshold
