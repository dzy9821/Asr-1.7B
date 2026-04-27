用 `multiprocessing.Pool + Queue` 处理音频的核心思路是：主进程通过 Queue 分发任务，子进程从 Queue 取任务处理，结果通过另一个 Queue 回传。但这个组合在 asyncio 环境下有一个经典陷阱需要注意。

先看完整代码示例：

```python
# src/services/vad_pool.py
import asyncio
import multiprocessing as mp
from multiprocessing import Pool, Queue
from typing import Any
import numpy as np

# ── 进程级全局变量，每个 worker 只初始化一次 ──────────────────────────
_vad_instance = None
_result_queue: Queue = None


def _init_worker(model_path: str, result_queue: Queue):
    """Pool initializer：子进程启动时执行一次，加载模型"""
    global _vad_instance, _result_queue
    # from src.models.vad import TenVAD
    # _vad_instance = TenVAD(model_path)
    _result_queue = result_queue
    print(f"[Worker {mp.current_process().pid}] VAD model loaded from {model_path}")


def _process_audio(task: dict) -> None:
    """
    Worker 执行函数：从 task 取音频，推理后把结果写入 result_queue。
    注意：返回值是 None，结果通过 Queue 异步回传，不走 Pool.apply_async 的回调。
    """
    task_id: str = task["task_id"]
    audio_bytes: bytes = task["audio"]

    try:
        # result = _vad_instance.process(audio_bytes)   # 真实调用
        # 模拟推理
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        result = {"task_id": task_id, "vad_segments": [{"start": 0, "end": len(audio_array)}], "error": None}
    except Exception as e:
        result = {"task_id": task_id, "vad_segments": None, "error": str(e)}

    _result_queue.put(result)  # 非阻塞写入


class VADPool:
    """
    封装 multiprocessing.Pool + Queue 的 VAD 推理池。

    架构说明：
      - task_queue  : 主进程 → worker，传递音频任务（此处直接用 Pool.apply_async 分发，
                      Pool 内部维护任务队列，无需额外 task_queue）
      - result_queue: worker → 主进程，回传推理结果
      - _pending    : 记录尚未收到结果的 task_id，用于 Future 映射
    """

    def __init__(self, num_workers: int, model_path: str):
        self._manager = mp.Manager()
        self._result_queue: Queue = self._manager.Queue()  # 跨进程安全 Queue
        self._pool = Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(model_path, self._result_queue),
        )
        self._pending: dict[str, asyncio.Future] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._consumer_task: asyncio.Task | None = None

    # ── 生命周期 ────────────────────────────────────────────────────────

    async def start(self):
        """在 asyncio 事件循环启动后调用，开启结果消费协程"""
        self._loop = asyncio.get_running_loop()
        self._consumer_task = asyncio.create_task(self._consume_results())

    async def shutdown(self):
        if self._consumer_task:
            self._consumer_task.cancel()
        self._pool.terminate()
        self._pool.join()

    # ── 核心接口 ────────────────────────────────────────────────────────

    async def process(self, task_id: str, audio_bytes: bytes) -> dict:
        """
        提交音频任务，返回 Future，await 后得到推理结果。
        Pool.apply_async 是非阻塞的，不会卡住事件循环。
        """
        future: asyncio.Future = self._loop.create_future()
        self._pending[task_id] = future

        task = {"task_id": task_id, "audio": audio_bytes}
        # apply_async 把任务丢给 Pool 内部队列，立即返回
        self._pool.apply_async(_process_audio, args=(task,))

        return await future  # 挂起当前协程，等待 _consume_results 设置结果

    # ── 内部结果消费循环 ────────────────────────────────────────────────

    async def _consume_results(self):
        """
        后台协程：轮询 result_queue，将结果分发给对应的 Future。

        注意：Queue.get() 是阻塞调用，必须用 run_in_executor 包装，
        否则会阻塞整个事件循环。
        """
        while True:
            try:
                result: dict = await self._loop.run_in_executor(
                    None,
                    self._result_queue.get  # 阻塞等待，但在线程池中执行
                )
                task_id = result["task_id"]
                future = self._pending.pop(task_id, None)

                if future and not future.done():
                    if result["error"]:
                        future.set_exception(RuntimeError(result["error"]))
                    else:
                        future.set_result(result)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[Consumer] Unexpected error: {e}")
```

然后是 FastAPI 生命周期集成和使用：

```python
# src/api/__init__.py
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket
from src.services.vad_pool import VADPool

vad_pool: VADPool = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vad_pool
    vad_pool = VADPool(num_workers=4, model_path="weights/ten-vad")
    await vad_pool.start()   # ← 必须在事件循环启动后调用
    yield
    await vad_pool.shutdown()


app = FastAPI(lifespan=lifespan)


@app.websocket("/tuling/asr/v3")
async def asr_endpoint(websocket: WebSocket):
    await websocket.accept()
    async for message in websocket.iter_json():
        audio_b64 = message["payload"]["audio"]["audio"]
        audio_bytes = __import__("base64").b64decode(audio_b64)

        task_id = str(uuid.uuid4())
        result = await vad_pool.process(task_id, audio_bytes)  # 非阻塞等待

        await websocket.send_json({"vad_result": result})
```

几个关键设计决策说明一下：

**为什么 `result_queue.get()` 要包在 `run_in_executor` 里？**
`Queue.get()` 是同步阻塞调用，如果直接 `await` 一个普通 Queue 会报错，直接调用又会冻结事件循环。用 `run_in_executor` 把它扔到线程池里阻塞等待，主线程的 asyncio 循环继续跑其他协程。

**为什么用 `Manager().Queue()` 而不是 `multiprocessing.Queue()`？**
`multiprocessing.Queue()` 直接传给 Pool 的 `initializer` 在某些平台（尤其是 `spawn` 启动方式，macOS/Windows 默认）会报 `can't pickle` 错误。`Manager().Queue()` 通过独立的 Manager 进程代理，序列化安全。

**`Pool.apply_async` 和 Queue 的分工**：Pool 的内部队列负责任务分发（保证 worker 负载均衡），`result_queue` 只负责结果回传，两者职责不重叠，代码更清晰。