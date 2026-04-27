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

优化方向应聚焦在：
- **实例池预热**：启动时预创建实例，避免首次连接的加载延迟（已实现）
- **实例回收复用**：连接关闭时销毁旧实例、创建新实例放回空闲池（已实现）
- **worker 数量调优**：通过 `VAD_WORKERS` 和 `VAD_CONNECTIONS_PER_INSTANCE` 控制总容量

TenVad 单实例内存极小（~306KB），创建开销约 0.5ms，无需为节省实例数量而牺牲正确性。