## 1. 项目概述

本项目是一个基于 WebSocket 的高并发、低延迟实时语音转录服务。客户端持续推送音频流，服务端通过级联模型管道完成语音活动检测（VAD）、语音识别（ASR）及文本后处理（ITN），并将结构化识别结果实时返回客户端。

**核心处理流程**：`音频流输入` → `VAD 语音断句` → `ASR 语音转文字` → `ITN 文本逆正则化` → `结构化结果输出`

**依赖模型与推理引擎**：

| 组件 | 模型/引擎 | 推理设备 | 说明 |
| :--- | :--- | :--- | :--- |
| **VAD** | ten-vad | CPU | 检测语音活动端点，实现精准断句；依赖原生 C 动态库（`libten_vad.so`），需在镜像内安装。 |
| **ASR** | Qwen/Qwen3-ASR-1.7B | **Ascend NPU** | 核心语音识别模型，基于 **vLLM-Ascend v0.18** 高性能推理；vLLM 服务与本推理服务共同打包进同一镜像，对外暴露 OpenAI 兼容 RESTful API，本服务通过 HTTP 调用该接口完成推理。 |
| **ITN** | fst_itn_zh | CPU | 逆文本正则化，如将"幺幺零"转换为"110"；依赖 Python 包 `WeTextProcessing`，需在镜像内安装。 |

---

## 2. 架构概览

### 2.1 核心架构模式

系统遵循分层架构设计，职责分离明确，便于测试与扩展：

- **API 层** (`src/api/`)：基于 FastAPI 的 WebSocket 网关，负责连接管理与协议握手。
- **服务层** (`src/services/`)：编排 VAD、ASR、ITN 核心业务逻辑。
- **模型层** (`src/models/`)：基于 Pydantic 定义的数据传输对象（DTO）与配置模型。
- **配置层**：基于 `pydantic-settings` 的环境变量集中管理。

**并发与隔离策略**：

- **异步 I/O**：WebSocket 连接处理使用 `asyncio` 支持高并发长连接。
- **进程池模式**：VAD 与 ITN 为 CPU 密集型任务，采用独立多进程池处理，避免阻塞事件循环。
- **vLLM 服务化**：ASR 推理由独立容器内的 vLLM 服务承载，本服务通过 OpenAI 兼容 RESTful API 与其通信，vLLM 进程常驻，无需每次请求重启。

### 2.2 数据流详解

1. **连接建立**：客户端连接至 `ws://host:port/tuling/asr/v3`，需在 **5 秒内**完成握手验证（携带 `traceId`、`appId`、`bizId`）。
2. **音频接收**：客户端持续发送 Base64 编码的 PCM 音频帧。
3. **语音活动检测**：VAD 服务实时分析音频流，检测话语结束（Endpoint）时触发回调。VAD 的 `hop_size` 固定为 **640 帧（40ms @ 16kHz）**，与客户端每 40ms 发送一帧音频的节律完全对齐，实现逐帧实时检测。**转写触发规则**：停顿等待时间随已收集语音长度线性缩短，语音越长、停顿越短即可触发转写。
4. **语音识别**：截取的完整语音片段通过 HTTP 请求发送至 vLLM 服务（OpenAI 兼容接口），由 Qwen3-ASR-1.7B 模型完成转写。热词通过拼接提示词（Prompt）的方式注入，以提升特定词汇识别准确率。
5. **文本后处理**：ASR 原始输出经 ITN 模型处理，转换为标准化文本（如数字、符号规范化）。
6. **结果推送**：服务端将带时间戳的词语级/句子级结果以 JSON 格式通过 WebSocket 推送给客户端。

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
- **数据校验**：Pydantic >=2.5.0 + pydantic-settings >=2.0.0
- **服务器**：Uvicorn（Standard）
- **可观测性**：prometheus-client

### 4.2 组件级额外依赖

| 组件 | 依赖 | 安装方式 | 说明 |
| :--- | :--- | :--- | :--- |
| **VAD** | `libten_vad.so` 原生 C 动态库 | 镜像构建阶段复制至系统库路径 | ten-vad 底层推理运行时，`models/vad/ten-vad/` 目录下已包含预编译产物 |
| **ITN** | `WeTextProcessing` | `pip install 'git+https://github.com/wenet-e2e/WeTextProcessing.git'` | 提供 `itn.chinese.inverse_normalizer.InverseNormalizer` |

### 4.3 开发与质量保障工具

| 类别 | 工具链 | 用途 |
| :--- | :--- | :--- |
| **测试** | pytest, pytest-asyncio, pytest-cov | 异步单元测试与覆盖率统计。 |
| **代码规范** | Ruff, Black, MyPy | 静态检查、格式化与类型校验。 |

---

## 5. 配置与部署运维

### 5.1 关键环境变量

| 变量名 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `WS_HOST` / `WS_PORT` | 0.0.0.0 / 8000 | 服务监听地址与端口。 |
| `VAD_WORKERS` | 16 | VAD CPU 密集型进程池大小，建议按宿主机 CPU 核数调整。 |
| `ITN_WORKERS` | 16 | ITN CPU 密集型进程池大小，建议按宿主机 CPU 核数调整。 |
| `MAX_CONNECTIONS` | 50 | 最大并发 WebSocket 连接数限制；超限时拒绝新连接（待确认：当前超限行为为拒绝还是排队，请补充说明）。 |
| `CUDA_VISIBLE_DEVICES` | 0 | 本服务可见的 GPU 设备 ID（注：GPU 资源主要由 vLLM 容器使用）。 |
| `LOG_LEVEL` | INFO | 日志输出级别（DEBUG 用于排查）。 |

### 5.2 模型权重管理

- **存储位置**：`./weights/`
- **包含内容**：
  - `fst_itn_zh/`：中文 ITN 模型文件。
  - `Qwen3-ASR-1.7B/`：ASR 模型权重及配置文件（由 vLLM 容器加载）。
- **容器化部署**：通过 Docker Volume 挂载至 `/weights` 目录。

### 5.3 常用运维命令

```bash
# 环境初始化
cp .env.example .env
uv sync

# 服务启动（开发模式）
uv run uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# 代码质量检查
uv run ruff check --fix . && uv run mypy src/

# 执行测试套件
uv run pytest --cov=src --cov-report=html

# Docker 部署
docker-compose up -d
```

### 5.4 镜像打包说明

完整推理服务（含 vLLM-Ascend 引擎与本推理服务）统一打包进同一镜像，**基础镜像为 `vllm-ascend:0.18`**。构建时需完成以下步骤：

1. **安装 VAD 原生 C 库**：将 `models/vad/ten-vad/` 下的预编译 `.so` 文件复制至镜像系统库路径（如 `/usr/local/lib`），并执行 `ldconfig`。
2. **安装 ITN Python 包**：
   ```bash
   pip install 'git+https://github.com/wenet-e2e/WeTextProcessing.git'
   ```
3. **复制项目代码**：将完整项目（含 `models/`、`src/`、`weights/` 等）COPY 进镜像。
4. **安装项目依赖**：`pip install -r requirements.txt`（或 `uv sync`）。
5. **设置启动命令**：镜像 CMD/ENTRYPOINT 负责同时启动 vLLM 服务（后台）与本推理服务（前台）。

> **注意**：模型权重（`weights/Qwen3-ASR-1.7B/` 等）体积较大，视需求决定是打进镜像还是以 Volume 形式挂载。推荐挂载方式以控制镜像体积。

---

## 6. 监控与可观测性

### 6.1 健康检查端点

| 端点 | 方法 | 描述 |
| :--- | :--- | :--- |
| `/api/v1/health` | GET | 服务进程存活检查。 |
| `/api/v1/ready` | GET | 模型加载就绪状态检查（用于 K8s Readiness Probe）。 |
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
| **握手超时（5s 断开）** | 检查客户端发送的 JSON 是否包含必填字段 `traceId`、`bizId`，以及 `header.status` 是否正确设为 `0`（握手帧）。 |
| **模型加载失败** | 执行 `curl http://localhost:8000/api/v1/ready` 查看状态；检查 `weights/` 目录挂载权限；确认 vLLM 容器已正常启动。 |
| **音频识别无结果** | 确认发送音频为 **PCM 16k/16bit** 格式，且 Base64 编码正确；检查 vLLM 服务是否可达（`curl $VLLM_API_BASE/models`）。 |
| **NPU 显存溢出（OOM）** | vLLM-Ascend 启动参数中调整 `--gpu-memory-utilization`；或降低 `MAX_CONNECTIONS` 以减少并发推理请求数。 |
| **识别延迟高** | 查看 `asr_queue_depth` 是否持续升高；检查 vLLM 侧延迟；考虑扩容 vLLM 实例或降低并发连接数。 |

### 7.1 调试模式开启

```bash
export LOG_LEVEL=DEBUG
uv run python main.py
```
