# 风险与待办事项

> 通过代码审查得出，按严重程度分 P0 / P1 / P2 三级，每项附具体文件位置、风险说明和修改建议。

---

## P0 — 高优先级（建议尽快修复）

### 1. VAD `_states` 存在 await 期间的竞态，可导致内存泄漏

**文件**：`src/services/vad_service.py:171-225`

**问题**：`_execute_batch` 在 `await asyncio.to_thread(...)` 之前读取了 `self._states[sid]` 的状态，在线程执行期间事件循环控制权被释放，此时若连接断开，`unregister_session(sid)` 会将 state 从 dict 中删除。随后 `_execute_batch` 恢复执行，又把 state 写回 `self._states`，形成僵尸 state 泄漏。

```python
# vad_service.py:200-218 问题片段
contexts = torch.cat([self._states[sid][0] for sid in valid_sids], dim=0)
states   = torch.cat([self._states[sid][1] for sid in valid_sids], dim=1)

# ★ 此 await 释放事件循环，unregister_session 可能在此期间执行
out, new_states = await asyncio.to_thread(self._raw_model, x, states)

# ★ 恢复执行后直接写入，未检查 sid 是否仍在 self._states 中
for i, (sid, future) in enumerate(zip(valid_sids, valid_futures)):
    self._states[sid] = (..., ...)  # ← 僵尸写入
```

**建议**：在写回 `self._states` 之前增加有效性检查：

```python
for i, (sid, future) in enumerate(zip(valid_sids, valid_futures)):
    if sid not in self._states:
        continue  # session 已注销，跳过
    self._states[sid] = (..., ...)
    future.set_result(out[i].item())
```

---

### 2. 连接信号量通过访问 CPython 私有属性实现

**文件**：`src/api/connection_manager.py:30-37`

**问题**：`try_acquire()` 直接读写 `asyncio.Semaphore._value`，这是 CPython 内部实现细节，不保证跨版本兼容。且在 Python 3.12+ 中 semaphore 的实现已有变化。`unregister` 调用 `release_slot()`（正常的 `semaphore.release()`），但如果有异常路径直接调用了 `release_slot` 而没经过正常的 acquire，可能导致计数溢出。

```python
# connection_manager.py:30-37
def try_acquire(self) -> bool:
    if self._semaphore._value <= 0:      # ← 访问私有属性
        return False
    self._semaphore._value -= 1          # ← 手动操作内部状态
    return True
```

**建议**：使用 `BoundedSemaphore` 配合显式的 `acquire` + try/except 模式，或改用 `asyncio.Lock` + 计数器：

```python
def try_acquire(self) -> bool:
    if self._semaphore.locked():         # 公开 API，安全
        return False
    # 使用内部队列 + 协程安全的 acquire_nowait 替代方案
    ...
```

---

### 3. VAD `feed_audio` 中 `np.concatenate` 每次全量拷贝

**文件**：`src/services/vad_service.py:283`

**问题**：每次喂入新音频都执行 `np.concatenate([self._sample_buffer, pcm_int16])`，对已有缓冲区做完整内存分配和拷贝。对于连续说话 30 秒的场景，缓冲区可累积约 960KB int16 数据，且每 32ms 触发一次完整拷贝，高频 GC 会导致延迟抖动。

```python
# vad_service.py:283
self._sample_buffer = np.concatenate([self._sample_buffer, pcm_int16])
```

**建议**：使用 list-of-chunks 模式，只在需要取帧时才 concat，或使用 ring buffer 预分配：

```python
# 方案 A: chunk list
self._chunks: list[np.ndarray] = []
self._chunk_total = 0

def feed_audio(self, pcm_int16):
    self._chunks.append(pcm_int16)
    self._chunk_total += len(pcm_int16)
    # 需要操作时将 chunks 拼接为 buffer，用完丢弃
    ...

# 方案 B: 预分配 ring buffer (如果上游帧大小固定)
```

---

## P1 — 中优先级（建议近期处理）

### 4. `session=None` 的控制流依赖隐性约定

**文件**：`src/api/websocket.py:75-77`

**问题**：`_handle_handshake` 返回 `None` 时，代码进入 `_wait_for_client_disconnect`——这是一个无限循环 `while True: await ws.receive_text()`，只在客户端断开时抛 `WebSocketDisconnect`。但此行为隐式假设该函数永不正常返回，如果将来任何人修改了这个函数（比如加了 timeout），`session` 仍为 `None` 但代码继续执行到 `session.sid`，直接 `AttributeError`。

```python
# websocket.py:75-80
session = await _handle_handshake(websocket)
if session is None:
    await _wait_for_client_disconnect(websocket)  # ← 无限循环或抛异常
# 下面这行在 session=None 时不可达，但完全依赖 _wait_for_client_disconnect 的实现
connection_manager.register(session.sid, session.trace_id)
```

**建议**：在 `if session is None` 块内显式 `return`，或抛一个明确异常：

```python
if session is None:
    await _wait_for_client_disconnect(websocket)
    return  # ← 显式终止，不依赖隐式行为
```

---

### 5. ITN 多进程池 shutdown 直接 hard kill

**文件**：`src/services/itn_pool.py:182-184`

**问题**：`pool.terminate()` 立即杀死所有 worker 进程，不等待正在执行的任务完成。在正常关闭流程中，如果有 ASR 任务刚拿到 vLLM 结果正准备调用 ITN，或 ITN worker 正在处理中，这些段的结果会丢失，客户端可能永远收不到最后几个 segment。

```python
# itn_pool.py:182-184
self._runtime.pool.terminate()   # ← 立即 SIGTERM，不等待
self._runtime.pool.join()
```

**建议**：先优雅关闭再 hard kill：

```python
self._runtime.pool.close()       # 阻止新任务提交
try:
    self._runtime.pool.join(timeout=5)  # 等待进行中的任务完成
except Exception:
    pass
self._runtime.pool.terminate()   # 超时后才强杀
self._runtime.pool.join()
```

---

### 6. 默认线程池被多处共用，存在资源争用

**文件**：
- `src/services/vad_service.py:211` — `asyncio.to_thread(self._raw_model, ...)`
- `src/services/asr_service.py:65` — `loop.run_in_executor(None, _encode_audio, ...)`

**问题**：VAD 模型推理和 ASR 音频编码都使用 Python 默认线程池（`min(32, cpu_count+4)` 个线程）。600 并发连接下，大量 VAD 批推理和 ASR 编码可能争抢同一批线程，导致其中一方等待线程可用。ITN 已经正确隔离了专用线程池（`itn_pool._executor`），但 VAD 侧没有。

**建议**：为 VAD 批处理器分配专用线程池：

```python
# vad_service.py
import concurrent.futures

class SileroVADBatchProcessor:
    def __init__(self, ...):
        ...
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="vad-infer",
        )

    async def _execute_batch(self, batch):
        ...
        out, new_states = await asyncio.get_event_loop().run_in_executor(
            self._executor, self._raw_model, x, states
        )
```

---

### 7. 无 WebSocket 消息大小限制

**文件**：`src/api/websocket.py:91`

**问题**：`await websocket.receive_text()` 没有任何消息大小限制。恶意或出错的客户端发送巨大的 Base64 音频帧时，服务端会尝试将整个消息加载到内存中解码。

```python
# websocket.py:91
raw = await websocket.receive_text()  # ← 无上限
```

**建议**：结合 uvicorn 的 `--ws-max-size` 参数限制，或在应用层校验：

```python
# uvicorn 启动时
uvicorn.run("main:app", ..., ws_max_size=2 * 1024 * 1024)  # 2MB 上限
```

---

### 8. ASR HTTP 客户端只重试连接错误，不重试 HTTP 错误状态码

**文件**：`src/services/asr_service.py:90-109`

**问题**：重试逻辑只捕获了 `ReadError / ConnectError / RemoteProtocolError`，但 vLLM 返回 429 (rate limit)、502 (bad gateway)、503 (overloaded) 时，`response.raise_for_status()` 会抛 `HTTPStatusError`，这个异常**没被捕获也不会重试**，直接向上传播到 `_process_segment` 被当作 ASR 推理失败处理。

**建议**：将 `HTTPStatusError`（至少 transient 状态码）加入重试逻辑：

```python
except (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError,
        httpx.HTTPStatusError) as exc:
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500:
        raise  # 4xx 客户端错误不重试
    # 5xx / 连接错误：重试
    ...
```

---

## P2 — 低优先级（技术债务，可在迭代中逐步清理）

### 9. `itn_service.py` 是死代码

**文件**：`src/services/itn_service.py`

**问题**：该文件实现了基于 `ThreadPoolExecutor` 的 ITN 服务，但全项目没有任何地方导入或使用它。实际生产使用的是 `itn_pool.py`（基于 multiprocessing）。

**建议**：删除 `src/services/itn_service.py`，减少维护负担。

---

### 10. `sys.path` 运行时注入

**文件**：`src/services/itn_pool.py:61-62`、`src/services/itn_service.py:31-32`

**问题**：两个文件都在 `import` 前动态往 `sys.path` 中插入模型目录。如果将来安装了同名的 pip 包或路径变更，import 行为不确定。

```python
models_dir = os.path.abspath(...)
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)
from itn_wrapper import ITNProcessor
```

**建议**：使用 `importlib` 按路径显式加载，或将模型目录做成 pip 包：

```python
import importlib.util
spec = importlib.util.spec_from_file_location("itn_wrapper", f"{models_dir}/itn_wrapper.py")
itn_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(itn_module)
```

---

### 11. `random.choices` 生成 session ID 存在碰撞风险

**文件**：`src/api/session.py:17-21`

**问题**：`AST_` + 13位随机字符（A-Z, 0-9），使用 `random.choices` 而非密码学安全随机。13 位 36 进制 = 36^13 ≈ 1.7×10^20 种可能，对于 600 并发来说碰撞概率极低，但 `random` 模块在 multiprocessing spawn 场景下可能因种子重复导致碰撞。

**建议**：改用 `secrets` 模块或 `uuid.uuid4().hex`：

```python
import secrets
import string

def _generate_sid() -> str:
    alphabet = string.ascii_uppercase + string.digits
    suffix = ''.join(secrets.choice(alphabet) for _ in range(13))
    return f"AST_{suffix}"
```

---

### 12. Metrics 和连接详情端点无鉴权

**文件**：`src/api/metrics.py:35`、`src/api/health.py:30-36`

**问题**：`/metrics` 和 `/api/v1/connections` 对外暴露，没有任何认证机制。`/connections` 直接返回所有活跃连接的 `sid → trace_id` 映射，可能泄露业务信息。

**建议**：至少对 `/connections` 增加简单的鉴权，或拆分为内部端口（如 9090）独立暴露 metrics/health 端点。

---

### 13. 配置文件硬编码内网 IP

**文件**：`src/core/config.py:23`

```python
VLLM_API_BASE: str = os.getenv("VLLM_API_BASE", "http://10.23.32.171:15002/v1")
```

**建议**：默认值改为空字符串或占位符，强制部署时通过环境变量显式设置。

---

### 14. 大二进制文件提交到 Git

**文件**：`120报警电话16k.wav`（2.6MB）

**问题**：测试用的 WAV 文件直接提交在仓库根目录，长期会增大 Git 历史体积。

**建议**：移到 `test/fixtures/` 目录下，或使用 Git LFS / 外部存储。

---

### 15. `client.java` 测试客户端混在项目根目录

**文件**：`client.java`

**问题**：Java 编写的并发测试客户端放在项目根目录，不属于 Python 服务代码，且包含硬编码的测试 URL。

**建议**：移到 `test/` 目录下的独立子目录。

---

### 16. `import json` 散落在函数体内

**文件**：`src/api/websocket.py:251, 394`

```python
# 第 251 行
import json as _json

# 第 394 行
import json
```

**建议**：统一移到文件顶部，避免每次调用重复 import（虽然 Python 会缓存，但影响可读性）。

---

## 附录：架构概览图

```
┌── 主进程 ───────────────────────────────────────────────┐
│  asyncio Event Loop（单线程）                             │
│  ├─ websocket_endpoint × N   每连接一个主协程              │
│  ├─ _process_segment × M     每语音段一个子协程（后台）      │
│  └─ _batch_loop × 1          全局 VAD 批处理协程           │
│                                                          │
│  默认线程池 (cpu_count+4)                                  │
│  ├─ asyncio.to_thread        VAD JIT 模型推理             │
│  └─ run_in_executor(None)    ASR 音频 WAV 编码            │
│                                                          │
│  ITN 提交线程池 (32 线程)                                  │
│  └─ run_in_executor(itn_executor)  阻塞等待子进程结果      │
│                                                          │
│  Daemon 线程 × 2                                          │
│  ├─ itn-dispatch            轮询 Manager.Queue 分发结果    │
│  └─ itn-queue-monitor       定期打印队列长度               │
│                                                          │
│  Manager 进程 × 1                                         │
│  └─ Manager().Queue()       跨进程结果传输                 │
└──────────────────────────────────────────────────────────┘
┌── ITN Worker 进程 × 8 (spawn) ──────────────────────────┐
│  ITNProcessor.process(text)  纯同步 CPU 操作              │
└──────────────────────────────────────────────────────────┘
```
