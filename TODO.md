# 风险与待办事项

> 通过代码审查 + README 交叉验证得出，按严重程度分 P0 / P1 / P2 三级。
> 标注 `[执行]` 的为确认待修复项；标注 `[执行]` 的为直接操作项。

---

## P0 — 高优先级（建议尽快修复）

### 0. ~~config.py 默认值改回 README 的生产默认值~~ ✅ 已完成

**文件**：`src/core/config.py:14-17` vs `README.md:198-201`

**背景**：代码中 600/60/300 是压测时改的测试参数，README 中 64/5/20 才是生产默认值。

| 变量 | config.py 当前默认 | README 生产默认 |
|------|-------------------|----------------|
| `MAX_CONNECTIONS` | 600 | **64** |
| `WS_PING_INTERVAL` | 60 | **5** |
| `WS_PING_TIMEOUT` | 300 | **20** |
| `VLLM_API_BASE` | `10.23.32.171:15002` | `148.148.52.127:15002` |

**操作**：

```python
# config.py —— 改回生产默认值
MAX_CONNECTIONS: int = int(os.getenv("MAX_CONNECTIONS", "64"))
WS_PING_INTERVAL: float = float(os.getenv("WS_PING_INTERVAL", "5"))
WS_PING_TIMEOUT: float = float(os.getenv("WS_PING_TIMEOUT", "20"))
```

压测时通过环境变量覆盖即可：`MAX_CONNECTIONS=600 WS_PING_INTERVAL=60 ...`

此外 README 列出 `pydantic-settings >=2.0.0` 但 `pyproject.toml` 中没有，确认是否需要补充。// 暂时不补充

---

### 1. ~~VAD `_states` 在 `await asyncio.to_thread` 期间可能被并发修改~~ ✅ 已完成

**文件**：`src/services/vad_service.py:171-225`

**问题**：`_execute_batch` 在 `await asyncio.to_thread(...)` 之前读取了 `self._states[sid]`，线程执行期间事件循环释放控制权。若此时 `unregister_session(sid)` 被调用，state 从 dict 中删除，之后 `_execute_batch` 恢复执行又写回 `self._states[sid]`，形成僵尸 state 泄漏。

**实际风险分析**：经过完整时序追溯，**正常断开路径下不会触发此竞态**。`feed_audio` 内部逐帧串行 `await future`，返回时所有 VAD future 均已 resolve，其后才走 `wait_pending_asr` → 发送终态 → 等客户端断开 → `finally` → `unregister_session`。此时 VAD 队列中已无该 session 的待处理帧。且 `WebSocketDisconnect` 只由 `receive_text()/send_text()` 抛出，不会在 `await future`（VAD future）期间触发。

**唯一可能场景**：应用 shutdown（容器 SIGTERM）期间，`_batch_loop` 正在 `asyncio.to_thread` 中处理包含该 session 的 batch，同时 uvicorn 触发连接关闭 → `finally` → `unregister_session`。竞态窗口 = VAD 推理耗时（3~8ms），概率极低。

**结论**：实际命中概率极低，但修复成本为零。保留作为防御性加固。

```python
# vad_service.py:200-218
contexts = torch.cat([self._states[sid][0] for sid in valid_sids], dim=0)
states   = torch.cat([self._states[sid][1] for sid in valid_sids], dim=1)

# ★ await 释放事件循环，unregister_session 可能在此期间执行
out, new_states = await asyncio.to_thread(self._raw_model, x, states)

# ★ 直接写回，未检查 sid 是否仍在 self._states 中
for i, (sid, future) in enumerate(zip(valid_sids, valid_futures)):
    self._states[sid] = (..., ...)  # ← 僵尸写入
```

**修复**：写回前检查 sid 有效性。

```python
for i, (sid, future) in enumerate(zip(valid_sids, valid_futures)):
    if sid not in self._states:
        continue
    self._states[sid] = (..., ...)
    future.set_result(out[i].item())
```

---

### 2. ~~VAD `feed_audio` 中 `np.concatenate` 每次全量内存拷贝~~ ✅ 已完成

**文件**：`src/services/vad_service.py:283`

**问题**：`np.ndarray` 是固定大小的连续内存块，没有 `.append()` 方法。每次调用 `np.concatenate` 都会分配全新内存 + 全量拷贝。对于连续说话 30s 场景（约 960KB int16），每 32ms 触发一次完整拷贝，高频 GC 导致延迟抖动。

```python
# vad_service.py:283 —— 每次都是完整分配 + 拷贝
self._sample_buffer = np.concatenate([self._sample_buffer, pcm_int16])
```

**修复**：用 Python list 收集 chunks（`.append` 是 O(1)），只在切帧时才临时 concat。

```python
def feed_audio(self, pcm_int16):
    self._chunks.append(pcm_int16)
    self._chunk_total += len(pcm_int16)

    # 当累积够一帧时，临时拼接 → 取帧 → 丢弃临时 buffer
    while self._chunk_total >= self.hop_size:
        buffer = np.concatenate(self._chunks)
        frame = buffer[:self.hop_size]
        remainder = buffer[self.hop_size:]
        self._chunks = [remainder] if len(remainder) > 0 else []
        self._chunk_total = len(remainder)
        ...
```

---

## P1 — 中优先级（建议近期处理）

### 3. ~~`session=None` 的控制流依赖隐性约定~~ ✅ 已完成

**文件**：`src/api/websocket.py:75-77`

**问题**：`_handle_handshake` 返回 `None` 时进入 `_wait_for_client_disconnect`（无限循环 `while True: await ws.receive_text()`），设计意图是等待客户端主动断开。但代码结构隐式依赖该函数永不正常返回——如果未来修改了 `_wait_for_client_disconnect`（比如加了 timeout），`session` 仍为 `None` 但代码会继续执行到 `session.sid`，直接 `AttributeError`。

```python
# websocket.py:75-80
session = await _handle_handshake(websocket)
if session is None:
    await _wait_for_client_disconnect(websocket)  # 无限循环，仅 WebSocketDisconnect 才退出
# 下行在 session=None 时不可达，但完全依赖上面函数永不正常返回
connection_manager.register(session.sid, session.trace_id)
```

**修复**：加显式 `return`，不改变行为，仅增加防御。

```python
if session is None:
    await _wait_for_client_disconnect(websocket)
    return
```

---

### 4. ~~ITN 多进程池 shutdown 直接 hard kill~~ ✅ 已完成

**文件**：`src/services/itn_pool.py:182-184`

**分析**：`itn_pool.shutdown()` 只在 FastAPI lifespan 的 shutdown 阶段调用（容器关闭时），正常连接断开不会触发。被 `terminate()` 影响的是"容器收到 SIGTERM 时还在 pipeline 中的少量段"。影响范围可控，但加短暂 grace period 成本极低。

```python
# itn_pool.py:182-184
self._runtime.pool.terminate()   # ← 立即 SIGTERM
self._runtime.pool.join()
```

**修复**：先 `close()` + 短暂 `join(timeout)`，超时后再 `terminate()`。

```python
self._runtime.pool.close()
try:
    self._runtime.pool.join(timeout=5)
except Exception:
    pass
self._runtime.pool.terminate()
self._runtime.pool.join()
```

---

### 5. ~~热词解析不支持 `|` 分隔符~~ ✅ 已完成

**文件**：`src/services/asr_service.py:149`

客户端以 `|` 为分隔符传热词（如 `张三丰|武当山|太极拳`），但 `build_hotword_context` 的 split 正则 `[,\n;，；、]+` 未包含 `|`，导致整个字符串被视为一个热词。

```python
# 当前：不拆分 |
words = [w.strip() for w in re.split(r"[,\n;，；、]+", hotwords) if w.strip()]
```

**修复**：将 `|` 加入分隔符。

```python
words = [w.strip() for w in re.split(r"[,\n;，；、|]+", hotwords) if w.strip()]
```

---

### 6. ~~ASR HTTP 客户端需向客户端透传 vLLM 错误码~~ ✅ 已完成

**文件**：`src/services/asr_service.py:90-109`、`src/api/websocket.py:340-351`

**问题**：vLLM 返回 429 / 502 / 503 等错误时，`response.raise_for_status()` 抛出 `HTTPStatusError`，当前未捕获（重试只覆盖 `ReadError/ConnectError/RemoteProtocolError`）。异常向上传播到 `_process_segment` 的 `except Exception` 分支，客户端只收到空结果 + generic "ASR error" 消息，无法区分是"识别失败"还是"vLLM 过载"。

**修复思路**：`asr_service.recognize()` 中捕获 `HTTPStatusError`，**不重试**（压测场景下重试加剧 vLLM 负载），而是将有意义的状态码和错误信息返回给调用方。同时确保 `asr_errors_total` 正确递增（当前已递增，此项 OK）。

具体方案：
1. `asr_service.recognize()` 捕获 `HTTPStatusError`，抛出带状态码的自定义异常
2. `_process_segment` 的 except 分支识别该异常，向客户端推送包含状态码的错误消息
3. 同时确保 `asr_errors_total` 正确计数（用于 Prometheus 计算 `asr_error_rate`）

---

## P2 — 低优先级（技术债务，逐步清理）

### 7. 连接信号量访问了私有属性 `_value` `[执行]`

**文件**：`src/api/connection_manager.py:30-37`

**分析**：CPython 3.11 中 `asyncio.Semaphore` 是纯 Python 实现，`try_acquire()` 没有 `await` 所以不存在运行时竞态。且 `_value` 属性名在 3.11/3.12/3.13 中未变化。**3.11 中功能安全，无运行时 bug。**风险纯粹是代码规范层面——访问私有属性违反封装，未来版本可能重构。

```python
def try_acquire(self) -> bool:
    if self._semaphore._value <= 0:      # ← 当前 3.11 中可工作
        return False
    self._semaphore._value -= 1
    return True
```

**修复**：用公开 API 重写，`locked()` 是 `asyncio.Semaphore` 的公开方法。

---

### 8. 删除死代码 `itn_service.py` `[执行]`

**文件**：`src/services/itn_service.py`

该文件实现了基于 `ThreadPoolExecutor` 的 ITN 服务，未被任何地方引用。生产使用的是 `itn_pool.py`。直接删除。

---

### 9. `sys.path` 运行时注入 `[执行]`

**文件**：`src/services/itn_pool.py:61-62`

```python
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)
from itn_wrapper import ITNProcessor
```

**修复**：用 `importlib.util.spec_from_file_location` 按路径显式加载。

---

### 10. 删除 `client.java` 测试客户端 `[执行]`

**文件**：`client.java`

Java 编写的并发测试客户端混在 Python 项目根目录，且包含硬编码测试 URL。删除。

---

### 11. `import json` 移到文件顶部 `[执行]`

**文件**：`src/api/websocket.py:251, 394`

```python
# 第 251 行
import json as _json
# 第 394 行
import json
```

移到文件顶部，统一为 `import json`。

---

### 12. README 移除 `asr_queue_depth` 指标，保留 `asr_error_rate` `[执行]`

**文件**：`README.md:266-268`

**分析**：
- `asr_queue_depth`：当前未实现，也不需要单独实现（Prometheus 可通过 pending task 数量间接反映）。从 README 中删除。
- `asr_error_rate`：保留。当前 `asr_errors_total` Counter（带 `error_type` 标签）已覆盖所有 ASR 异常路径（handshake_timeout / internal / asr_inference），包括 vLLM HTTP 错误码（通过 `_process_segment` 的 `except Exception` 分支递增）。在 Prometheus 中通过 `rate(asr_errors_total[5m])` 即可得到错误率。

**操作**：README §6.2 中删除 `asr_queue_depth` 行，保留 `asr_error_rate` 行。

---

### 13. URL 从 `/tuling/asr/v3` 改为 `/tuling/ast/v3` `[执行]`

**文件**：`src/api/websocket.py:59`、`README.md:38, 72`

```python
# websocket.py:59
@router.websocket("/tuling/ast/v3")
```

同步更新 README 中的端点描述。

---

### 14. 默认端口改为 8856，简化 README 启动命令 `[执行]`

**文件**：`src/core/config.py:13`、`README.md:196, 230, 284, 294`

**config.py**：
```python
WS_PORT: int = int(os.getenv("WS_PORT", "8856"))
```

**README 改动**：
- 配置表 `WS_PORT` 默认值：8000 → 8856
- 启动命令简化（`python main.py` 自动读取 config 中所有参数，无需 CLI 传参）：
  ```bash
  # 改为
  python main.py
  ```
- 调试命令同理简化，curl 示例中端口改为 8856

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
