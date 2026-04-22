import argparse
import json
import os
import queue
import sys
import threading
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import multiprocessing as mp
import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DEFAULT_AUDIO_PATH = "/home/ubuntu/project/ASR-1.7B/Asr-1.7B/120报警电话16k.wav"

# worker 进程内全局变量
_SESSIONS: dict[str, object] = {}
_RESULT_QUEUE = None


def _init_worker(result_queue):
    global _RESULT_QUEUE, _SESSIONS
    _RESULT_QUEUE = result_queue
    _SESSIONS = {}


def _serialize_segments(segments: list[dict]) -> list[dict]:
    serialized = []
    for seg in segments:
        serialized.append(
            {
                "start_sample": int(seg["start_sample"]),
                "end_sample": int(seg["end_sample"]),
                "audio_len": int(len(seg["audio"])),
            }
        )
    return serialized


def _worker_process_task(task: dict) -> None:
    from src.services.vad_service import StreamingVADSession

    task_id = task["task_id"]
    op = task["op"]
    session_id = task["session_id"]
    sample_rate = int(task.get("sample_rate", 16000))
    hop_size = int(task.get("hop_size", 640))

    try:
        session = _SESSIONS.get(session_id)
        if session is None and op in {"feed", "flush"}:
            session = StreamingVADSession(sample_rate=sample_rate, hop_size=hop_size)
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
                        "audio_len": int(len(seg["audio"])),
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


@dataclass
class _WorkerRuntime:
    pool: Any
    result_queue: object
    pending: dict[str, queue.Queue]
    pending_lock: threading.Lock
    dispatcher: threading.Thread
    running: threading.Event


class VADDesignPoolQueueRouter:
    """
    test-only 实验实现：
    - 使用 Queue + 多进程
    - 按 session_id 做固定路由，确保同一会话始终落在同一 worker
    """

    def __init__(self, num_workers: int = 2):
        if num_workers <= 0:
            raise ValueError("num_workers must be > 0")
        self._ctx = mp.get_context("spawn")
        self._manager = self._ctx.Manager()
        self._workers: list[_WorkerRuntime] = []
        self._num_workers = num_workers

        for _ in range(num_workers):
            result_q = self._manager.Queue()
            pool = self._ctx.Pool(
                processes=1,
                initializer=_init_worker,
                initargs=(result_q,),
            )
            runtime = _WorkerRuntime(
                pool=pool,
                result_queue=result_q,
                pending={},
                pending_lock=threading.Lock(),
                dispatcher=threading.Thread(),
                running=threading.Event(),
            )
            runtime.running.set()
            runtime.dispatcher = threading.Thread(
                target=self._dispatch_results_loop,
                args=(runtime,),
                daemon=True,
            )
            runtime.dispatcher.start()
            self._workers.append(runtime)

    def close(self) -> None:
        for worker in self._workers:
            worker.running.clear()
        for worker in self._workers:
            try:
                worker.pool.terminate()
                worker.pool.join()
            except Exception:
                pass
        try:
            self._manager.shutdown()
        except Exception:
            pass

    def _dispatch_results_loop(self, runtime: _WorkerRuntime) -> None:
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

    def _select_worker_index(self, session_id: str) -> int:
        # 固定路由，避免会话状态在多个进程间分裂。
        return hash(session_id) % self._num_workers

    def _submit(self, session_id: str, task: dict, timeout_sec: float = 30.0) -> dict:
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
            raise TimeoutError(f"task timeout: {task_id}") from exc

        if not result.get("ok"):
            raise RuntimeError(result.get("error", "unknown worker error"))
        return result["payload"]

    def feed_audio(
        self,
        session_id: str,
        pcm_int16: np.ndarray,
        sample_rate: int = 16000,
        hop_size: int = 640,
    ) -> list[dict]:
        payload = self._submit(
            session_id=session_id,
            task={
                "op": "feed",
                "session_id": session_id,
                "audio_bytes": pcm_int16.tobytes(),
                "sample_rate": sample_rate,
                "hop_size": hop_size,
            },
        )
        return payload["segments"]

    def flush(self, session_id: str, sample_rate: int = 16000, hop_size: int = 640) -> dict | None:
        payload = self._submit(
            session_id=session_id,
            task={
                "op": "flush",
                "session_id": session_id,
                "sample_rate": sample_rate,
                "hop_size": hop_size,
            },
        )
        return payload["segment"]

    def close_session(self, session_id: str) -> None:
        self._submit(session_id=session_id, task={"op": "close", "session_id": session_id})


def load_wav(file_path: str) -> tuple[np.ndarray, int]:
    with wave.open(file_path, "rb") as wf:
        if wf.getnchannels() != 1:
            raise ValueError(f"Only mono wav supported, got {wf.getnchannels()} channels")
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        audio = wf.readframes(n_frames)
    return np.frombuffer(audio, dtype=np.int16), sample_rate


def _to_timestamps(segments: list[dict], sample_rate: int) -> list[tuple[float, float]]:
    return [
        (round(seg["start_sample"] / sample_rate, 3), round(seg["end_sample"] / sample_rate, 3))
        for seg in segments
    ]


def run_connection(
    router: VADDesignPoolQueueRouter,
    session_id: str,
    audio_data: np.ndarray,
    sample_rate: int,
    chunk_size: int = 640,
) -> dict:
    timestamps: list[tuple[float, float]] = []
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i : i + chunk_size]
        if len(chunk) < chunk_size:
            break
        segments = router.feed_audio(session_id, chunk, sample_rate=sample_rate, hop_size=chunk_size)
        timestamps.extend(_to_timestamps(segments, sample_rate))

    flush_seg = router.flush(session_id, sample_rate=sample_rate, hop_size=chunk_size)
    if flush_seg is not None:
        timestamps.extend(_to_timestamps([flush_seg], sample_rate))
    router.close_session(session_id)

    return {
        "session_id": session_id,
        "segment_count": len(timestamps),
        "timestamps": timestamps,
        "first_5_timestamps": timestamps[:5],
        "last_5_timestamps": timestamps[-5:] if len(timestamps) > 5 else timestamps,
    }


def run_experiment(
    audio_path: str,
    num_workers: int = 2,
    chunk_size: int = 640,
) -> dict:
    audio_data, sample_rate = load_wav(audio_path)
    router = VADDesignPoolQueueRouter(num_workers=num_workers)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            fut1 = executor.submit(
                run_connection,
                router,
                "session-A",
                audio_data,
                sample_rate,
                chunk_size,
            )
            fut2 = executor.submit(
                run_connection,
                router,
                "session-B",
                audio_data,
                sample_rate,
                chunk_size,
            )
            result_a = fut1.result()
            result_b = fut2.result()
    finally:
        router.close()

    return {
        "audio_path": audio_path,
        "sample_rate": sample_rate,
        "chunk_size": chunk_size,
        "num_workers": num_workers,
        "connection_A": result_a,
        "connection_B": result_b,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="VAD design experiment: queue + multiprocessing in test scope")
    parser.add_argument("--audio", default=DEFAULT_AUDIO_PATH)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=640)
    args = parser.parse_args()

    result = run_experiment(
        audio_path=args.audio,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def test_vad_design_pool_queue_manual():
    import pytest

    if os.getenv("RUN_VAD_DESIGN_EXPERIMENT") != "1":
        pytest.skip("manual experiment; set RUN_VAD_DESIGN_EXPERIMENT=1 to run")
    if not os.path.exists(DEFAULT_AUDIO_PATH):
        pytest.skip(f"audio file not found: {DEFAULT_AUDIO_PATH}")
    result = run_experiment(DEFAULT_AUDIO_PATH, num_workers=2, chunk_size=640)
    assert result["connection_A"]["segment_count"] >= 0
    assert result["connection_B"]["segment_count"] >= 0


if __name__ == "__main__":
    raise SystemExit(main())
