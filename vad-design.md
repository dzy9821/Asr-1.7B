# VAD 多进程池设计文档

## 1. 架构概述

VAD（语音活动检测）服务采用 **多进程池 + Queue** 架构，将 CPU 密集型的 C 运行时推理隔离在独立进程中，主进程的 asyncio 事件循环不会被阻塞。

### 核心设计决策

| 决策 | 选择 | 理由 |
|:---|:---|:---|
| 进程启动模式 | `spawn` | 避免 fork 后 C 动态库（`libten_vad.so`）状态异常 |
| 任务分发 | `Pool.apply_async` | Pool 内部维护任务队列，简洁可靠 |
| 结果回传 | `Manager().Queue()` | 跨进程安全序列化，兼容 `spawn` 模式 |
| 会话路由 | `hash(session_id) % num_workers` | 固定路由确保同一会话的帧始终在同一进程处理 |
| 初始化时机 | 服务启动时 eager init | 避免首次请求的冷启动延迟 |

### 架构图

```
                         ┌──────────────────────────────────────────────────────┐
                         │                  主进程 (asyncio)                     │
                         │                                                      │
                         │   WebSocket Handler                                  │
                         │     │                                                │
                         │     ▼                                                │
                         │   VADPool                                            │
                         │     │ hash(session_id) % N                           │
                         │     ├─────────────────┐                              │
                         │     ▼                 ▼                              │
                         │   Worker-0          Worker-1        ...  Worker-N    │
                         │   ┌──────────┐      ┌──────────┐                    │
                         │   │ dispatch │      │ dispatch │   (dispatcher线程) │
                         │   │ thread   │      │ thread   │                    │
                         │   └────┬─────┘      └────┬─────┘                    │
                         │        │                  │                          │
                         └────────┼──────────────────┼──────────────────────────┘
                                  │                  │
                    ┌─────────────┼──────────────────┼─────────────────────┐
                    │  子进程层    │                  │                     │
                    │             ▼                  ▼                     │
                    │   ┌──────────────────┐ ┌──────────────────┐         │
                    │   │  Process-0       │ │  Process-1       │  ...    │
                    │   │                  │ │                  │         │
                    │   │ _SESSIONS = {    │ │ _SESSIONS = {    │         │
                    │   │   sid_A: VAD,    │ │   sid_B: VAD,    │         │
                    │   │   sid_C: VAD,    │ │   sid_D: VAD,    │         │
                    │   │ }                │ │ }                │         │
                    │   │                  │ │                  │         │
                    │   │ → result_queue   │ │ → result_queue   │         │
                    └──────────────────┘ └──────────────────┘         │
                    └─────────────────────────────────────────────────────┘
```

## 2. 生命周期

### 2.1 启动流程

```python
# main.py lifespan
vad_pool.start()   # 创建 N 个 spawn 子进程，每个进程初始化 _SESSIONS={}, _RESULT_QUEUE
itn_pool.start()   # 创建 M 个 spawn 子进程，每个进程预加载 ITNProcessor
await asr_service.startup()  # 初始化 httpx 客户端
```

启动时 VAD 子进程**不预加载模型**，`StreamingVADSession` 在子进程收到第一个 `feed` 或 `flush` 任务时按需创建。

### 2.2 请求处理流程

1. **WebSocket 收到音频帧** → `vad_pool.feed_audio(session_id, pcm_int16)`
2. **feed_audio** 通过 `run_in_executor` 将同步 `_submit()` 丢到线程池
3. **_submit** 做 hash 路由选择 worker → `pool.apply_async` 提交任务
4. **子进程** 执行 `_worker_process_task`：查找/创建 session → 调用 `feed_audio` → 序列化结果 → 写入 `_RESULT_QUEUE`
5. **dispatcher 线程** 从 `result_queue` 取出结果 → 找到对应的 `waiter` Queue → 通知主线程
6. **_submit** 从 `waiter.get()` 拿到结果 → 返回 → `run_in_executor` 完成 → 协程恢复

### 2.3 关闭流程

```python
# main.py lifespan shutdown
await asr_service.shutdown()
itn_pool.shutdown()   # terminate + join
vad_pool.shutdown()   # terminate + join + manager.shutdown
```

## 3. 关键实现细节

### 3.1 为什么用 `Manager().Queue()` 而不是 `multiprocessing.Queue()`？

`multiprocessing.Queue()` 直接传给 Pool 的 `initializer` 在 `spawn` 启动方式下会报 `can't pickle` 错误。`Manager().Queue()` 通过独立的 Manager 进程代理，序列化安全。

### 3.2 为什么 `result_queue.get()` 要包在 `run_in_executor` 里？

在异步接口 `feed_audio` / `flush` 中，底层 `_submit()` 是同步阻塞调用（等待 `waiter.get()`）。必须用 `run_in_executor` 将其丢到线程池执行，否则会冻结 asyncio 事件循环。

### 3.3 `Pool.apply_async` 和 Queue 的分工

- `Pool.apply_async`：负责将任务分发给 Pool 内部的 worker（Pool 自带任务队列）
- `result_queue`：负责结果回传（worker → 主进程）
- 两者职责不重叠，代码更清晰

### 3.4 音频数据的序列化

`numpy.ndarray` 不能直接通过 multiprocessing 传递。发送端先 `.tobytes()` 转为 `bytes`，接收端再 `np.frombuffer(..., dtype=np.int16).copy()` 还原。`copy()` 确保返回的数组拥有独立内存。

### 3.5 会话固定路由

```python
def _select_worker_index(self, session_id: str) -> int:
    return hash(session_id) % self._num_workers
```

同一 `session_id` 的所有 `feed` / `flush` / `close` 操作始终路由到同一 worker 进程。这是因为 `StreamingVADSession` 有内部状态（帧缓冲、语音/静默计数器），不能跨进程分裂。

## 4. ITN 多进程池

ITN 池与 VAD 池架构类似，但更简单：

- **无需会话路由**：ITN 是无状态的文本处理，任何进程都可以处理任何请求
- **Pool 内部负载均衡**：直接利用 `Pool.apply_async` 的自动任务分发
- **启动时预热**：initializer 中预加载 `ITNProcessor`，首次请求无冷启动延迟
- **经实测**：多进程可有效降低多并发时的 ITN 处理延迟，优于线程池（GIL 限制）

## 5. 配置参数

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `VAD_WORKERS` | 32 | VAD 进程池实例数 |
| `VAD_CONNECTIONS_PER_INSTANCE` | 2 | 每实例承载连接数上限 |
| `ITN_WORKERS` | 8 | ITN 进程池实例数 |
| `MAX_CONNECTIONS` | 64 | 最大并发连接 = `VAD_WORKERS * VAD_CONNECTIONS_PER_INSTANCE` |

> **约束**：`MAX_CONNECTIONS` 始终为 `VAD_WORKERS` 的整倍数。调整 VAD 实例数时需同步更新。

## 6. 文件清单

| 文件 | 职责 |
|:---|:---|
| `src/services/vad_service.py` | `StreamingVADSession` 类（子进程内使用），动态阈值判定逻辑 |
| `src/services/vad_pool.py` | `VADPool` 多进程池路由器（主进程使用），生命周期管理 |
| `src/services/itn_pool.py` | `ITNPool` 多进程池（主进程使用），生命周期管理 |
| `src/api/websocket.py` | 全局 `vad_pool` / `itn_pool` 实例，WebSocket handler 调用入口 |
| `main.py` | lifespan 启动/关闭 pool |