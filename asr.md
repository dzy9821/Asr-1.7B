## 1. 项目概述

本项目是一个基于 WebSocket 的高并发、低延迟实时语音转录服务。客户端持续推送音频流，服务端通过级联模型管道完成语音活动检测（VAD）、语音识别（ASR）及文本后处理（ITN），并将结构化识别结果实时返回客户端。

**核心处理流程**：`音频流输入` → `VAD 语音断句` → `ASR 语音转文字` → `ITN 文本逆正则化` → `结构化结果输出`

**依赖模型与推理引擎**：

| 组件 | 模型/引擎 | 推理设备 | 说明 |
| :--- | :--- | :--- | :--- |
| **VAD** | Silero VAD (v5, JIT) | CPU | 检测语音活动端点，实现精准断句；基于 PyTorch JIT 模型，支持 **动态批处理**（单实例同时服务所有并发连接）。 |
| **ASR** | Qwen/Qwen3-ASR-1.7B | **Ascend NPU** | 核心语音识别模型，基于 **vLLM-Ascend v0.18** 高性能推理；vLLM 服务与本推理服务共同打包进同一镜像，对外暴露 OpenAI 兼容 RESTful API，本服务通过 HTTP 调用该接口完成推理。 |
| **ITN** | fst_itn_zh | CPU | 逆文本正则化，如将"幺幺零"转换为"110"；依赖 Python 包 `WeTextProcessing`，需在镜像内安装。 |

---

## 2. 架构概览

### 2.1 核心架构模式

系统遵循分层架构设计，职责分离明确，便于测试与扩展：

- **API 层** (`src/api/`)：基于 FastAPI 的 WebSocket 网关，负责连接管理与协议握手。
- **服务层** (`src/services/`)：编排 VAD、ASR、ITN 核心业务逻辑。
- **模型层** (`src/models/`)：基于 Pydantic 定义的数据传输对象（DTO）与配置模型。
- **配置层**：基于环境变量的集中管理，代码内提供默认值。

**并发与隔离策略**：

- **异步 I/O**：WebSocket 连接处理使用 `asyncio` 支持高并发长连接。
- **VAD 动态批处理**：采用 Silero VAD **全局单实例 + 动态批处理** 架构。服务启动时加载一个 Silero JIT 模型（`SileroVADBatchProcessor` 全局单例），每个连接的时序状态（RNN context + hidden state）在服务端外部管理。后台 asyncio Task 从队列中收集多个连接的帧请求，凑批后统一执行一次 batch forward，再将概率结果分发给各连接。经压测验证，单实例 + batch_size=256 可在实时约束内服务 **500 路并发**（RTF < 1.0）。每个连接持有独立的 `StreamingVADSession`，负责帧缓冲与动态阈值断句逻辑。
- **ITN 多进程池**：ITN 采用固定 **8 个多进程实例**（`spawn` 模式），请求通过 Pool 内部队列自动负载均衡分发。每个进程预加载 `ITNProcessor` 单例，避免 GIL 限制下的 CPU 密集型 FST 计算瓶颈。结果通过 `Manager().Queue()` 跨进程安全回传。服务启动时 **eager init** 所有进程并预热模型。
- **vLLM 服务化**：ASR 推理由独立容器内的 vLLM 服务承载，本服务通过 OpenAI 兼容 RESTful API 与其通信，vLLM 进程常驻，无需每次请求重启。

### 2.2 数据流详解

1. **连接建立**：客户端连接至 `ws://host:port/tuling/asr/v3`，需在 **5 秒内**完成握手验证。系统最大并发连接数为 **64**。WebSocket 连接依赖底层的 Ping/Pong 机制维持（建议设置 `ping_interval=20`，`ping_timeout=300` 以适应高并发）。
2. **音频接收**：客户端持续发送 Base64 编码的 PCM 音频帧。
3. **语音活动检测**：VAD 服务实时分析音频流，检测话语结束（Endpoint）时触发回调。Silero VAD 的 `window_size` 固定为 **512 samples（32ms @ 16kHz）**。客户端发送的音频在服务端内部被重新切片为 512-sample 帧后提交至全局批处理器。每个连接持有独立的 `StreamingVADSession` 实例，保证时序状态隔离（状态由批处理器外部管理）。
   - **动态转写触发规则**：停顿等待时间随已收集语音长度线性缩短，具体逻辑如下：
     - **短音频抑制**：语音时长 `< 0.5s`，视为噪声或误触，不触发（继续等待）。
     - **长音频强制触发**：语音时长 `>= 30.0s`，无论停顿多久立即触发，防止缓冲区堆积（即转发给 ASR 的语音最长为 30 秒）。
     - **动态停顿阈值（0\~20s）**：累积语音 `0s` 时需停顿 `2.0s (T_MAX)` 方可触发；累积至 `20s` 时仅需 `0.5s (T_MIN)`，两者之间线性递减（斜率 `K = 0.075`）。语音越长，触发断句需要的停顿越短。
     - **固定停顿阈值（20\~30s）**：累积语音 `>= 20s` 后，停顿 `0.5s` 即触发转发。
4. **语音识别**：截取的完整语音片段通过 HTTP 请求发送至 vLLM 服务（OpenAI 兼容接口），由 Qwen3-ASR-1.7B 模型完成转写。热词通过拼接提示词（Prompt）的方式注入，以提升特定词汇识别准确率。
5. **文本后处理**：ASR 原始输出经 ITN 模型处理，转换为标准化文本（如数字、符号规范化）。ITN 请求通过 8 实例多进程池自动负载均衡分流。
6. **结果推送**：服务端将带时间戳的词语级/句子级结果以 JSON 格式通过 WebSocket 推送给客户端。

### 2.3 场景示例：三段式连续语音处理流

假设客户端持续录音并推送，用户完整表达了三句话，整个流式处理的时序如下：

1. **握手与推流开始**：客户端发送 `status: 0` 帧建立连接。随后开始以 40ms 间隔持续推送音频流帧（`status: 1`）。
2. **第一段（"你好"） —— 正常停顿触发**：
   - **输入**：用户说了 1s 的"你好"，然后思考停顿了 2.0s。
   - **VAD 判定**：当前收集语音长 1s，根据公式计算动态停顿阈值约为 `2.0 - 0.075*1 = 1.925s`。实际停顿 2.0s `>` 1.925s，**成功触发第一次断句**。
   - **ASR 与推送**：服务端截取这 1s 的音频封装成 HTTP 请求发给 vLLM；拿到 ASR 结果并经过 ITN 后，通过 WebSocket 将"你好"推送给客户端（附带 `segId: 0` 和识别中状态 `status: 1`）。此时 VAD 缓冲区清空并重新开始收集。
3. **第二段（"帮我查一下今天的天气"） —— 动态缩短阈值触发**：
   - **输入**：用户说了 6s（语速较慢），然后轻微停顿了 1.6s。
   - **VAD 判定**：当前语音长 6s，动态停顿阈值随之降低，约为 `2.0 - 0.075*6 = 1.55s`。实际停顿 1.6s `>` 1.55s，**成功触发第二次断句**。
   - **ASR 与推送**：截取这 6s 音频请求 vLLM，随后推送结果"帮我查一下今天的天气"（附带 `segId: 1` 和识别中状态 `status: 1`）。VAD 缓冲区再次清空。
4. **第三段（"特别是下午会不会下雨"） —— 客户端主动结束触发**：
   - **输入**：用户最后说了 3s 的内容。说完后，用户立刻松开语音按钮或关闭麦克风，客户端发送结束帧（`status: 2`）。
   - **VAD 判定**：服务端收到 `status: 2`，无视当前是否达到停顿阈值，**强制截断并触发最后一段的转写**。
   - **ASR 与推送**：截取最后 3s 音频请求 vLLM。处理完成后，向客户端推送最终结果"特别是下午会不会下雨"（附带 `segId: 2` 和整体结束状态 `status: 2`）。结果发送完毕后，**服务端不会主动断开 WebSocket 连接**，而是等待客户端主动关闭，以防止提前断连导致结果未送达或状态异常。

---

## 3. 接口定义（WebSocket API）

- **端点**：`ws(wss)://[ip]:[port]/tuling/asr/v3`

### 3.1 请求参数结构

客户端发送的 JSON 消息结构如下：

```json
{
    "header": {
        "traceId": "traceId123456",
        "appId": "123456",
        "bizId": "39769795890",
        "status": 0
    },
    "payload": {
        "audio": {
            "audio": "JiuY3iK9AAB..."
        },
        "text": {
            "text": "张三疯"
        }
    }
}
```

**字段约束说明**：

| 字段路径 | 类型 | 必填 | 描述 |
| :--- | :--- | :--- | :--- |
| `header.traceId` | String | **是** | 全链路日志追踪标识。 |
| `header.appId` | String | 否 | 调用方应用标识。 |
| `header.bizId` | String | **是** | 业务唯一标识，通常对应用户 ID。 |
| `header.status` | Int | **是** | **客户端帧类型**：`0` 握手帧（首帧）；`1` 音频数据帧；`2` 结束帧。 |
| `header.resIdList` | List[String] | 否 | 多用户场景下的辅助 ID 列表。 |
| `payload.audio.audio` | String | **是** | Base64 编码的音频数据（PCM 16k/16bit 格式）。 |
| `payload.text.text` | String | 否 | 用于提升识别准确率的热词，将在服务端以提示词拼接方式注入 ASR 推理。 |

### 3.2 响应结果结构

服务端推送的识别结果 JSON 示例：

```json
{
  "header": {
    "code": 0,
    "message": "success",
    "sid": "AST_MKMZO0WX2SLZ4",
    "traceId": "traceId123456",
    "status": 1
  },
  "payload": {
    "result": {
      "segId": 0,
      "bg": 140,
      "ed": 3230,
      "msgtype": "sentence",
      "ws": [
        {
          "bg": 17,
          "cw": [{
            "w": "你好",
            "wp": "n",
            "wb": 17,
            "we": 56,
            "sc": "0.00",
            "sf": 0,
            "wc": "0.00"
          }]
        }
      ]
    }
  }
}
```

**关键字段语义说明**：

| 字段路径 | 类型 | 描述 |
| :--- | :--- | :--- |
| `header.status` | Int | **识别进度状态**：`0` 开始；`1` 识别进行中；`2` 识别结束（终态）。 |
| `header.sid` | String | 当前 WebSocket 会话的唯一标识。 |
| `payload.result.bg` / `ed` | Int | 句子级起始/结束时间，单位：**毫秒（ms）**。 |
| `payload.result.ws[].cw[].w` | String | 识别出的具体词汇。 |
| `payload.result.ws[].cw[].wb` / `we` | Int | 词汇级起始/结束时间，单位：**10 毫秒（即厘秒，cs）**，与句子级单位不同。 |

---

## 4. 技术栈与依赖管理

### 4.1 运行环境与核心库

- **Python 版本**：3.11.14
- **Web 框架**：FastAPI + websockets ^12.0
- **推理引擎**：**vLLM-Ascend v0.18**（与本推理服务打包于同一镜像，提供 OpenAI 兼容 API）
- **数据处理**：NumPy ==1.26.4
- **VAD 推理**：PyTorch >=2.0.0 + torchaudio >=2.0.0（Silero VAD JIT 模型依赖）
- **数据校验**：Pydantic >=2.5.0 + pydantic-settings >=2.0.0
- **服务器**：Uvicorn（Standard）
- **可观测性**：prometheus-client

### 4.2 组件级额外依赖

| 组件 | 依赖 | 安装方式 | 说明 |
| :--- | :--- | :--- | :--- |
| **VAD** | Silero VAD JIT 模型 (`silero_vad.jit`) | 已包含在项目 `models/vad/silero-vad/` 目录 | 基于 PyTorch JIT 的 VAD 模型，通过 `torch.hub.load(source='local')` 加载，无需额外安装原生库 |
| **ITN** | `WeTextProcessing` | `pip install 'git+https://github.com/wenet-e2e/WeTextProcessing.git'` | 提供 `itn.chinese.inverse_normalizer.InverseNormalizer` |

### 4.3 开发与质量保障工具

| 类别 | 工具链 | 用途 |
| :--- | :--- | :--- |
| **测试** | pytest, pytest-asyncio, pytest-cov | 异步单元测试与覆盖率统计。 |
| **代码规范** | Ruff, Black, MyPy | 静态检查、格式化与类型校验。 |

---

## 5. 配置与部署运维

### 5.1 关键环境变量

> 所有配置均在代码内提供默认值，通过 `os.getenv()` 读取同名环境变量覆盖。无需任何配置文件，未来在 `docker-compose.yaml` 的 `environment` 段中直接注入即可。

| 变量名 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `WS_HOST` / `WS_PORT` | 0.0.0.0 / 8000 | 服务监听地址与端口。 |
| `ITN_WORKERS` | 8 | ITN 多进程池实例数（固定容量设计，`spawn` 模式）。每个进程预加载 `ITNProcessor` 单例。 |
| `MAX_CONNECTIONS` | 64 | 最大并发 WebSocket 连接数限制，超限时**直接拒绝**新连接（WebSocket close code `1013 Try Again Later`）。 |
| `HANDSHAKE_TIMEOUT` | 5 | 握手超时时间（秒），连接建立后须在此时间内完成首帧验证。 |
| `WS_PING_INTERVAL` | 20 | WebSocket 心跳（Ping）发送间隔（秒）。 |
| `WS_PING_TIMEOUT` | 300 | WebSocket 心跳超时时间（秒）。高并发下设置较长以避免误判掉线。 |
| `MP_QUEUE_LOG_INTERVAL_SEC` | 10 | ITN 多进程池队列深度监控日志打印间隔（秒）。 |
| `VLLM_API_BASE` | http://148.148.52.127:15002/v1 | vLLM 服务的 OpenAI 兼容 API 地址。 |
| `VLLM_MODEL_NAME` | Qwen3-ASR-1.7B | vLLM 中加载的 ASR 模型名称。 |
| `VLLM_API_KEY` | EMPTY | vLLM API 密钥（默认无鉴权）。 |
| `ASCEND_RT_VISIBLE_DEVICES` | 0 | 本服务可见的 Ascend NPU 设备 ID（由 vLLM-Ascend 使用）。 |
| `HOTWORDS` | （空） | 服务端默认热词列表，逗号分隔（如 `张三丰,武当山,太极拳`）。客户端传入的热词会追加合并。 |
| `LOG_LEVEL` | INFO | 日志输出级别（DEBUG 用于排查）。 |

### 5.2 模型权重管理

- **存储位置**：`./weights/`
- **包含内容**：
  - `fst_itn_zh/`：中文 ITN 模型文件。
  - `Qwen3-ASR-1.7B/`：ASR 模型权重及配置文件（由 vLLM-Ascend 加载）。
- **容器化部署**：ASR 模型权重体积较大，**必须通过 Docker Volume 挂载**至容器内 `/weights` 目录，不打入镜像。

### 5.3 常用运维命令

```bash
# 依赖安装
pip install -r requirements.txt

# 服务启动
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --ws-ping-interval 20 --ws-ping-timeout 300

# Docker 部署
docker-compose -f docker-compose.yaml up -d
```

### 5.4 镜像打包说明

完整推理服务（含 vLLM-Ascend 引擎与本推理服务）统一打包进同一镜像，**基础镜像为 `vllm-ascend:0.18`**。构建时需完成以下步骤：

1. **安装 ITN Python 包**：
   ```bash
   pip install 'git+https://github.com/wenet-e2e/WeTextProcessing.git'
   ```
2. **复制项目代码**：将完整项目（含 `models/`、`src/` 等）COPY 进镜像。Silero VAD JIT 模型已包含在 `models/vad/silero-vad/` 目录中，无需额外安装原生库。**注意**：`weights/` 目录不 COPY 进镜像，通过 Volume 挂载。
3. **安装项目依赖**：`pip install -r requirements.txt`（或 `uv sync`）。PyTorch 如已在基础镜像中预装则无需重复安装。
4. **设置启动命令**：用户手动执行 vLLM 服务与本推理服务的启动命令。

> **注意**：ASR 模型权重（`weights/Qwen3-ASR-1.7B/`）**必须以 Volume 形式挂载**，不打入镜像，以控制镜像体积并方便模型版本更新。

---

## 6. 监控与可观测性

### 6.1 健康检查端点

| 端点 | 方法 | 描述 |
| :--- | :--- | :--- |
| `/api/v1/health` | GET | 服务进程存活检查。 |
| `/api/v1/ready` | GET | 模型加载就绪状态检查（用于 K8s Readiness Probe）。复用全局 `asr_service` 实例。 |
| `/api/v1/connections` | GET | 当前活跃连接数统计。 |
| `/metrics` | GET | Prometheus 格式指标暴露。 |

### 6.2 关键性能指标（KPIs）

- `asr_connections_current`：实时连接数 Gauge。
- `asr_processing_latency_ms`：ASR 处理延迟直方图（含 vLLM HTTP 调用耗时）。
- `asr_queue_depth`：任务积压队列长度。
- `asr_error_rate`：接口错误率。

### 6.3 日志规范

- **格式**：JSON 结构化日志，输出至 `stdout`。
- **关联字段**：每条日志均包含 `trace_id`，支持全链路追踪。
- **敏感信息**：日志中不记录完整的 Base64 音频数据。

---

## 7. 故障排除指南

| 常见问题 | 排查步骤与解决方案 |
| :--- | :--- |
| **WebSocket 频繁断开 (1006)** | 高并发下事件循环偶发阻塞，导致 Ping 超时。启动时需加上 `--ws-ping-timeout 300`。 |
| **握手超时（5s 断开）** | 检查客户端发送的 JSON 是否包含必填字段 `traceId`、`bizId`，以及 `header.status` 是否正确设为 `0`（握手帧）。 |
| **模型加载失败** | 执行 `curl http://localhost:8000/api/v1/ready` 查看状态；检查 `weights/` 目录挂载权限；确认 vLLM 容器已正常启动。 |
| **音频识别无结果** | 确认发送音频为 **PCM 16k/16bit** 格式，且 Base64 编码正确；检查 vLLM 服务是否可达（`curl $VLLM_API_BASE/models`）。 |
| **NPU 显存溢出（OOM）** | vLLM-Ascend 启动参数中调整 `--gpu-memory-utilization`；或降低 `MAX_CONNECTIONS` 以减少并发推理请求数。 |
| **识别延迟高** | 查看 `asr_queue_depth` 是否持续升高；检查 vLLM 侧延迟；考虑扩容 vLLM 实例或降低并发连接数。 |
| **ITN 进程池启动慢** | 使用 `spawn` 模式的多进程池启动较慢属正常现象（需序列化并重新加载模型）。启动期间服务不可用，Readiness Probe 会返回 `not_ready`。 |

### 7.1 调试模式开启

```bash
export LOG_LEVEL=DEBUG
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --ws-ping-interval 20 --ws-ping-timeout 300
```

---

## 8. 项目进度追踪

> 最后更新时间：2026-04-27

### 已完成

- [x] 系统架构设计文档（本文档）
- [x] 项目骨架搭建（`src/core/`、`src/api/`、`src/services/`、`src/utils/`、`src/models/`）
- [x] 全局配置管理（`config.py`，环境变量驱动，代码内默认值，无配置文件依赖）
- [x] JSON 结构化日志 + trace_id 上下文注入（`logging.py`）
- [x] 请求/响应 Pydantic 数据模型（`schemas.py`，对齐 §3 接口协议）
- [x] 流式 VAD 断句服务（`vad_service.py`，Silero VAD 动态批处理 + 动态停顿阈值 + 强制触发）
- [x] VAD 全局批处理器（`vad_service.py`，`SileroVADBatchProcessor` 单例 + 外部状态管理 + asyncio 凑批推理）
- [x] 异步 ASR 推理服务（`asr_service.py`，httpx → vLLM OpenAI 兼容接口）
- [x] ITN 多进程池（`itn_pool.py`，8 实例 spawn 模式 + Pool 内部负载均衡 + Queue 结果回传 + eager init 预热）
- [x] WebSocket 全链路处理（`websocket.py`，握手→音频→断句→推理→推送）
- [x] 并发连接管理（`connection_manager.py`，Semaphore + 1013 拒绝）
- [x] 会话状态机（`session.py`，sid 生成、seg_id 递增；每 Session 注册至 VAD 批处理器，close() 时注销）
- [x] 健康探针与 Prometheus 指标暴露（`health.py`、`metrics.py`）
- [x] 虚拟环境与依赖安装（含 WeTextProcessing + PyTorch），本地启动验证通过
- [x] **[P0]** 端到端联调：连接远程 vLLM ASR，跑通完整管线
- [x] **[P0]** VAD 流式断句验证：用真实音频验证 Silero VAD 断句准确性与时间戳

### 待完成

- [x] **[P1]** WebSocket 测试客户端脚本与并发压测脚本 (`ws_stress_test.py` / `client.java`)
- [ ] **[P1]** docker-compose.yaml：vLLM-Ascend 容器编排 + 环境变量注入
- [ ] **[P1]** Dockerfile：镜像打包（代码+依赖），权重 Volume 挂载
- [x] **[P2]** 异常恢复与重试策略 (vLLM 断连自动重试，阻塞操作移出 Event Loop)
- [x] **[P2]** 多并发压力测试 (服务端不主动断开连接，避免读写冲突)
- [ ] **[P2]** 单元测试覆盖
- [ ] **[P3]** Grafana 监控面板
