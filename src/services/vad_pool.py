"""
VAD 多进程池 —— 基于 multiprocessing.Pool + Manager().Queue() 的生产实现。

架构说明：
  - 使用 spawn 模式创建 N 个 worker 进程，每个进程内部维护若干 StreamingVADSession
  - 按 session_id hash 做固定路由，确保同一会话始终落在同一 worker
  - 结果通过 Manager().Queue() 跨进程安全回传
  - 主进程端通过 dispatcher 线程轮询结果队列，分发给对应的 waiter

生命周期：
  - 应用启动时调用 start()，预创建所有进程池（eager init）
  - 应用关闭时调用 shutdown()，终止所有进程

参考：test/test_vad_design_pool_queue_experiment.py
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import multiprocessing as mp
import queue
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)

# ============================================================
# Worker 进程内全局变量（每个进程独立副本）
# ============================================================

_SESSIONS: dict[str, object] = {}
_RESULT_QUEUE: Any = None


def _safe_qsize(queue_obj: Any) -> Optional[int]:
    """Best-effort 读取队列长度；不支持时返回 None。"""
    try:
        return int(queue_obj.qsize())
    except Exception:
        return None


def _init_worker(result_queue: Any) -> None:
    """Pool initializer：子进程启动时执行一次。"""
    global _RESULT_QUEUE, _SESSIONS
    _RESULT_QUEUE = result_queue
    _SESSIONS = {}
    logger.info("VAD worker process initialized: pid=%d", mp.current_process().pid)


def _serialize_segments(segments: list[dict]) -> list[dict]:
    """将 VAD 返回的段序列化（去除 numpy 数组，保留元信息）。"""
    serialized = []
    for seg in segments:
        serialized.append(
            {
                "start_sample": int(seg["start_sample"]),
                "end_sample": int(seg["end_sample"]),
                "audio_bytes": seg["audio"].tobytes(),
            }
        )
    return serialized


def _worker_process_task(task: dict) -> None:
    """
    Worker 执行函数：根据 op 执行 feed/flush/close。

    结果通过 _RESULT_QUEUE 回传，不走 Pool.apply_async 的返回值。
    """
    from src.services.vad_service import StreamingVADSession

    task_id = task["task_id"]
    op = task["op"]
    session_id = task["session_id"]
    sample_rate = int(task.get("sample_rate", 16000))
    hop_size = int(task.get("hop_size", 640))

    try:
        session = _SESSIONS.get(session_id)
        if session is None and op in {"feed", "flush"}:
            session = StreamingVADSession(
                sample_rate=sample_rate, hop_size=hop_size
            )
            _SESSIONS[session_id] = session

        if op == "feed":
            pcm_bytes = task["audio_bytes"]
            pcm = np.frombuffer(pcm_bytes, dtype=np.int16).copy()
            segments = session.feed_audio(pcm)
            payload = {"segments": _serialize_segments(segments)}

        elif op == "flush":
            seg = session.flush()
            payload = {
                "segment": (
                    None
                    if seg is None
                    else {
                        "start_sample": int(seg["start_sample"]),
                        "end_sample": int(seg["end_sample"]),
                        "audio_bytes": seg["audio"].tobytes(),
                    }
                )
            }

        elif op == "close":
            _SESSIONS.pop(session_id, None)
            payload = {"closed": True}

        else:
            raise ValueError(f"unknown op: {op}")

        _RESULT_QUEUE.put({"task_id": task_id, "ok": True, "payload": payload})

    except Exception as exc:
        _RESULT_QUEUE.put({"task_id": task_id, "ok": False, "error": repr(exc)})


# ============================================================
# Worker 运行时描述
# ============================================================


@dataclass
class _WorkerRuntime:
    pool: Any
    result_queue: Any
    pending: dict[str, queue.Queue] = field(default_factory=dict)
    pending_lock: threading.Lock = field(default_factory=threading.Lock)
    dispatcher: Optional[threading.Thread] = None
    running: threading.Event = field(default_factory=threading.Event)


# ============================================================
# VADPool —— 生产级多进程池路由器
# ============================================================


class VADPool:
    """
    VAD 多进程池路由器。

    - num_workers 个进程，每个进程维护独立的 StreamingVADSession 实例
    - 按 session_id hash 做固定路由，确保同一会话的所有帧都在同一进程处理
    - 每个 worker 最多服务 connections_per_instance 个并发连接
    - 总承载量 = num_workers * connections_per_instance
    """

    def __init__(
        self,
        num_workers: int | None = None,
        connections_per_instance: int | None = None,
    ):
        self._num_workers = num_workers or settings.VAD_WORKERS
        self._connections_per_instance = (
            connections_per_instance or settings.VAD_CONNECTIONS_PER_INSTANCE
        )

        if self._num_workers <= 0:
            raise ValueError("num_workers must be > 0")

        self._ctx = mp.get_context("spawn")
        self._manager: Any = None
        self._workers: list[_WorkerRuntime] = []
        self._monitor_interval_sec = max(0.0, settings.MP_QUEUE_LOG_INTERVAL_SEC)
        self._monitor_running = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        # 独立线程池，避免与 ITN / ASR 编码共用默认 executor 导致线程饥饿
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self._num_workers * self._connections_per_instance,
            thread_name_prefix="vad-submit",
        )

    def start(self) -> None:
        """
        启动所有 worker 进程（应用启动时调用，eager init）。
        """
        self._manager = self._ctx.Manager()

        for i in range(self._num_workers):
            result_q = self._manager.Queue()
            pool = self._ctx.Pool(
                processes=1,
                initializer=_init_worker,
                initargs=(result_q,),
            )
            runtime = _WorkerRuntime(
                pool=pool,
                result_queue=result_q,
            )
            runtime.running.set()
            runtime.dispatcher = threading.Thread(
                target=self._dispatch_results_loop,
                args=(runtime,),
                daemon=True,
                name=f"vad-dispatch-{i}",
            )
            runtime.dispatcher.start()
            self._workers.append(runtime)

        if self._monitor_interval_sec > 0:
            self._monitor_running.clear()
            self._monitor_thread = threading.Thread(
                target=self._log_queue_stats_loop,
                daemon=True,
                name="vad-queue-monitor",
            )
            self._monitor_thread.start()

        logger.info(
            "VAD pool started: %d workers, %d conn/instance, max_capacity=%d",
            self._num_workers,
            self._connections_per_instance,
            self._num_workers * self._connections_per_instance,
        )

    def shutdown(self) -> None:
        """终止所有 worker 进程并释放资源。"""
        self._monitor_running.set()
        for worker in self._workers:
            worker.running.clear()
        for worker in self._workers:
            try:
                worker.pool.terminate()
                worker.pool.join()
            except Exception:
                pass
        try:
            if self._manager:
                self._manager.shutdown()
        except Exception:
            pass
        self._monitor_thread = None
        self._workers.clear()
        logger.info("VAD pool shutdown complete")

    # ---- 结果分发循环 ----

    def _dispatch_results_loop(self, runtime: _WorkerRuntime) -> None:
        """后台线程：轮询 result_queue，将结果分发给对应的 waiter。"""
        while runtime.running.is_set():
            try:
                result = runtime.result_queue.get(timeout=0.2)
            except Exception:
                continue
            task_id = result.get("task_id")
            if not task_id:
                continue
            with runtime.pending_lock:
                waiter = runtime.pending.pop(task_id, None)
            if waiter is not None:
                waiter.put(result)

    def _log_queue_stats_loop(self) -> None:
        """后台线程：定期打印队列与待回包任务长度。"""
        while not self._monitor_running.wait(timeout=self._monitor_interval_sec):
            if not self._workers:
                continue

            pending_total = 0
            result_queue_total = 0
            result_queue_known = True
            active_workers = 0
            max_pending = -1
            max_pending_worker = -1
            max_result_queue = -1
            max_result_queue_worker = -1

            for idx, runtime in enumerate(self._workers):
                with runtime.pending_lock:
                    pending_size = len(runtime.pending)
                pending_total += pending_size
                if pending_size > 0:
                    active_workers += 1
                if pending_size > max_pending:
                    max_pending = pending_size
                    max_pending_worker = idx

                result_qsize = _safe_qsize(runtime.result_queue)
                if result_qsize is None:
                    result_queue_known = False
                else:
                    result_queue_total += result_qsize
                    if result_qsize > max_result_queue:
                        max_result_queue = result_qsize
                        max_result_queue_worker = idx

            result_queue_total_text = (
                str(result_queue_total) if result_queue_known else "unknown"
            )
            max_result_queue_text = (
                f"{max_result_queue}@worker-{max_result_queue_worker}"
                if result_queue_known
                else "unknown"
            )

            logger.info(
                "VAD queue stats: pending_total=%d active_workers=%d/%d max_pending=%d@worker-%d result_queue_total=%s max_result_queue=%s",
                pending_total,
                active_workers,
                self._num_workers,
                max_pending,
                max_pending_worker,
                result_queue_total_text,
                max_result_queue_text,
            )

    # ---- 路由 ----

    def _select_worker_index(self, session_id: str) -> int:
        """按 session_id hash 做固定路由。"""
        return hash(session_id) % self._num_workers

    # ---- 同步提交 ----

    def _submit(
        self, session_id: str, task: dict, timeout_sec: float = 30.0
    ) -> dict:
        """提交任务并等待结果（同步阻塞）。"""
        idx = self._select_worker_index(session_id)
        runtime = self._workers[idx]

        task_id = uuid.uuid4().hex
        task["task_id"] = task_id

        waiter: queue.Queue = queue.Queue(maxsize=1)
        with runtime.pending_lock:
            runtime.pending[task_id] = waiter

        runtime.pool.apply_async(_worker_process_task, args=(task,))

        try:
            result = waiter.get(timeout=timeout_sec)
        except Exception as exc:
            with runtime.pending_lock:
                runtime.pending.pop(task_id, None)
            raise TimeoutError(f"VAD task timeout: {task_id}") from exc

        if not result.get("ok"):
            raise RuntimeError(result.get("error", "unknown VAD worker error"))
        return result["payload"]

    # ---- 异步接口（供 WebSocket handler 调用） ----

    async def feed_audio(
        self,
        session_id: str,
        pcm_int16: np.ndarray,
        sample_rate: int = 16000,
        hop_size: int = 640,
    ) -> list[dict]:
        """
        异步喂入音频帧。

        Returns:
            触发的语音段列表，每项为
            {"audio": np.ndarray (int16), "start_sample": int, "end_sample": int}
        """
        loop = asyncio.get_running_loop()
        payload = await loop.run_in_executor(
            self._executor,
            self._submit,
            session_id,
            {
                "op": "feed",
                "session_id": session_id,
                "audio_bytes": pcm_int16.tobytes(),
                "sample_rate": sample_rate,
                "hop_size": hop_size,
            },
        )
        # 将 bytes 还原为 numpy
        segments = []
        for seg in payload["segments"]:
            segments.append(
                {
                    "audio": np.frombuffer(
                        seg["audio_bytes"], dtype=np.int16
                    ).copy(),
                    "start_sample": seg["start_sample"],
                    "end_sample": seg["end_sample"],
                }
            )
        return segments

    async def flush(
        self,
        session_id: str,
        sample_rate: int = 16000,
        hop_size: int = 640,
    ) -> Optional[dict]:
        """异步刷出残余语音段。"""
        loop = asyncio.get_running_loop()
        payload = await loop.run_in_executor(
            self._executor,
            self._submit,
            session_id,
            {
                "op": "flush",
                "session_id": session_id,
                "sample_rate": sample_rate,
                "hop_size": hop_size,
            },
        )
        seg = payload["segment"]
        if seg is None:
            return None
        return {
            "audio": np.frombuffer(seg["audio_bytes"], dtype=np.int16).copy(),
            "start_sample": seg["start_sample"],
            "end_sample": seg["end_sample"],
        }

    async def close_session(self, session_id: str) -> None:
        """异步关闭会话（释放子进程端的 VAD 实例）。"""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._executor,
            self._submit,
            session_id,
            {"op": "close", "session_id": session_id},
        )

    @property
    def num_workers(self) -> int:
        return self._num_workers

    @property
    def max_capacity(self) -> int:
        return self._num_workers * self._connections_per_instance
