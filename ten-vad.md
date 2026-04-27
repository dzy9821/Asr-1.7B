# VAD 共享实例可行性分析

## 背景

目标：探索在一个 worker 进程下，用一个 TenVad 实例同时服务 2 个 WebSocket 连接，以减少内存和实例数量。

## 结论

**❌ 不可行。** 一个 TenVad 实例无法同时服务两个音频流。

---

## 实验验证

测试脚本：[test_vad_shared_instance.py](test/test_vad_shared_instance.py)

### 实验1：状态污染检测（独占 vs 共享交替）

将同一段音频（120报警电话16k.wav, 2030帧）分成两个流，对比独占实例 vs 共享实例交替处理的 flag 输出：

| 场景 | 流A flag 差异 | 流B flag 差异 |
|------|-------------|-------------|
| 独占 vs 共享交替 | **423/1015 (41.7%)** | **258/1015 (25.4%)** |

**41.7% 的帧 flag 不一致**，说明 TenVad 内部维护了时序隐藏状态（类似 RNN/GRU 的 hidden state），交替喂入不同流的帧会彻底污染这些状态。

差异示例：
```
帧   1 (  0.04s): 独占flag=0 prob=0.327 | 共享flag=1 prob=0.506
帧  10 (  0.40s): 独占flag=0 prob=0.070 | 共享flag=1 prob=0.726
帧  14 (  0.56s): 独占flag=0 prob=0.194 | 共享flag=1 prob=0.804
```

### 实验2：逐帧丢弃上下文（per-frame create/destroy）

每帧 `ten_vad_create → process → destroy`，模拟无状态处理：

| 指标 | 持久实例 | 逐帧新建 |
|------|---------|---------|
| 耗时(500帧) | 169ms (0.338ms/帧) | 281ms (0.563ms/帧) |
| 性能倍数 | 1x | **1.7x** |
| flag 差异 | baseline | **45/500 (9.0%)** |
| probability 平均偏差 | - | **0.150** |

- 性能开销增加 70%，尚在可接受范围
- 但 **9% 的帧判定错误**，probability 平均偏差 0.15（满值 1.0），无法用于生产
- 丢弃上下文后，模型在语音段内频繁将语音帧误判为静默（prob 从 0.9+ 掉到 0.4），导致 VAD 端点检测完全失效

### 实验3：线程安全性

同一 TenVad 实例在两个线程中同时调用 `process()`：
- **直接抛出 AssertionError**（C 底层数据竞争导致帧长度校验失败）
- TenVad 的 C 运行时 **非线程安全**

### 实验4：串行复用（流A完成后接流B）

同一实例先处理完流A全部帧，再处理流B：
- **触发 double free / core dump** — C 层内存管理出错
- 说明 TenVad 实例不支持在多个生命周期间安全复用

---

## 根因分析

TenVad 的 C API 设计：

```c
ten_vad_create(handle, hop_size, threshold);   // 创建带内部状态的不透明句柄
ten_vad_process(handle, audio_data, ...);       // 每帧推理，依赖历史 hidden state
ten_vad_destroy(handle);                        // 销毁实例
```

关键限制：
1. **有状态推理**：`process()` 的输出依赖之前所有帧累积的隐藏状态（时序模型特征）
2. **无 reset API**：没有 `ten_vad_reset()` 来清空内部状态
3. **非线程安全**：C 运行时不支持并发访问
4. **实例不可复用**：连续处理不同流会导致内存错误

## 架构建议

当前架构（每连接一个独立 `StreamingVADSession` 实例）是 **正确且必要的** 设计：

```
Worker 进程
├── Session A → TenVad 实例 A（独立 handle + 独立 hidden state）
└── Session B → TenVad 实例 B（独立 handle + 独立 hidden state）
```

## 架构变更记录

### 2026-04-27：从多进程池改为连接级实例

**变更前**：`vad_pool.py` 使用 `multiprocessing.Pool`（32 worker，每 worker 服务 2 连接），按 `session_id` hash 路由，通过 `Manager().Queue()` 回传结果。

**变更后**：删除 `vad_pool.py`，在 `ASRSession.__init__` 中直接创建 `StreamingVADSession` 实例，连接关闭时随 Session 销毁。

**决策依据**：
1. TenVad 内部维护时序隐藏状态，实例间不可共享（实验1：41.7% flag 污染）
2. 逐帧丢弃上下文也不可行（实验2：9% flag 错误，probability 偏差 0.15）
3. TenVad 非线程安全（实验3：AssertionError）
4. 实例不可跨生命周期复用（实验4：double free crash）
5. TenVad 极轻量（~306KB/实例，创建 ~0.5ms），多进程池的管理开销远大于实例本身