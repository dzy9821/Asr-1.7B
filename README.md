# ASR Real-Time Streaming Service (ASR-1.7B)

本项目是一个基于 WebSocket 的高并发、低延迟实时语音识别（ASR）流式服务。该服务充当了客户端和后端推理大模型（部署于 vLLM）之间的网关层与业务管线，实现了实时断句、异步转写、文本逆正则化（ITN）以及全链路指标监控。

## 核心架构概览

服务核心业务管线为：
`WebSocket 接收音频` -> `逐帧 VAD 实时断句` -> `按段调用 vLLM ASR 识别` -> `ITN 文本逆正则化` -> `WebSocket 推送结果`。

## 目录结构与文件职责详细说明

### 1. 根目录文件
*   **`main.py`**: 整个 FastAPI 服务的启动入口。负责注册各个 HTTP/WebSocket 路由（API 端点），并管理全局生命周期（在启动和关闭时初始化/销毁 ASR 服务与 ITN 线程池）。启动命令：`uvicorn main:app --host 0.0.0.0 --port 8000`。
*   **`pyproject.toml`**: 项目的依赖与元数据配置文件，使用 `uv` 或标准的 Python 打包工具链管理。
*   **`docker-compose.yaml`**: 服务的容器编排配置。推荐通过该文件的 `environment` 字段统一注入所需的各种运行环境变量。
*   **`asr.md`**: 项目的系统级架构设计文档，包含了接口协议规范、动态断句规则、环境配置等详尽的设计说明。

### 2. 核心模块 (`src/core/`)
该模块提供项目的基础公共设施。
*   **`config.py`**: 极简的全局统一配置管理。所有参数（端口、并发上限、vLLM 地址、热词等）均通过 `os.getenv()` 读取环境变量，代码内提供默认值。无需任何配置文件，未来通过 `docker-compose.yaml` 的 `environment` 段注入即可。支持的环境变量包括 `HOTWORDS`（服务端默认热词，逗号分隔，客户端传入的热词会追加合并）等。
*   **`logging.py`**: 日志配置模块。配置全局日志输出为标准输出（stdout）并采用结构化的 JSON 格式。最重要的是，利用 `contextvars` 实现了在异步环境下的 `trace_id` 注入，方便整条链路的日志追踪。

### 3. API 与网关层 (`src/api/`)
该模块直接负责与客户端的 HTTP/WebSocket 交互及连接状态维护。
*   **`websocket.py`**: **系统的核心枢纽**。实现了 `/tuling/asr/v3` WebSocket 接口，负责处理连接的握手超时控制、接收客户端持续发送的 PCM 音频、调度 VAD/ASR/ITN 核心管线，以及按照约定的 JSON 协议把识别结果推回客户端。
*   **`session.py`**: 抽象了单个客户端连接的会话上下文 (`ASRSession`)。为每个连接维护了一个独立的状态机、随机生成的 13 位会话 ID (`sid`)、段序号 (`seg_id`) 的递增逻辑、以及绑定的专属 `StreamingVADSession`。
*   **`connection_manager.py`**: 全局连接管理器。内部使用异步信号量 (`asyncio.Semaphore`) 控制最大并发连接数。超出并发上限的连接将被立刻拒绝（返回 1013）。同时维护着一个活跃连接注册表。
*   **`health.py`**: 提供基础的系统探针，包含应用存活探针 (`/health`)、大模型链路就绪探针 (`/ready`) 以及当前的连接概览 (`/connections`)，方便 K8s 或其他监控系统的探测。
*   **`metrics.py`**: 暴露出 Prometheus 规范的指标端点 (`/metrics`)。实时监控当前并发数、管线处理延迟（Histogram）、总处理段数与各类异常错误统计。

### 4. 服务管线层 (`src/services/`)
该模块将底层的算法模型包装成了适合在异步高并发 Web 服务中运行的组件。
*   **`vad_service.py`**: 流式语音活动检测服务。封装了底层的 TenVAD。它的核心职责是维持 `StreamingVADSession`，将客户端传入的不定长音频切片攒成标准的 40ms 帧送入 VAD 引擎。此外，它还内置了**动态停顿阈值策略**（语音越长，触发断句所需的停顿越短），在识别准确度和延迟间取得平衡。
*   **`asr_service.py`**: 异步语音识别服务。利用 `httpx.AsyncClient` 负责将 VAD 切割出来的话语片段，异步提交给后端独立部署的 vLLM OpenAI 兼容接口，并负责从响应中提取与清理识别文本，处理大模型的异常。
*   **`itn_service.py`**: 文本逆正则化（数字、符号转换）服务。因为 ITN 操作是纯 CPU 密集型任务，如果在异步主线程运行会导致阻塞，所以这里通过包装一个专门的 `ThreadPoolExecutor` (线程池)，把同步的 ITN 处理逻辑安全地融入到 `asyncio` 的事件循环中。

### 5. 工具与数据结构 (`src/utils/` & `src/models/`)
*   **`utils/audio.py`**: 音频数据处理的纯函数工具箱。主要负责将客户端发来的 Base64 编码数据解码为 `numpy` 的 int16 数组格式，以及进行采样点到毫秒/厘秒的时间维度换算。
*   **`models/schemas.py`**: 定义了客户端请求和系统响应的 Pydantic 数据模型。确保系统接收和发出的 JSON 结构严格遵守 `asr.md` 中定义的接口协议，具备完善的类型校验与 IDE 提示支持。

### 6. 底层模型依赖 (`models/`)
*(该部分通常复用现有的算法工程结构)*
*   **`vad/`**: 底层基于 C/C++ 编译的 TenVAD 运行时及其 Python wrapper（如 `ten_vad_wrapper.py`）。
*   **`itn/`**: 基于 WeTextProcessing 的中文逆文本正则化逻辑与 FST 模型。

---

## 项目进度

### ✅ 已完成

| 模块 | 完成内容 |
|:--|:--|
| **架构设计** | 完成系统架构文档 `asr.md`，含接口协议、动态断句规则、环境变量、三段式处理流程示例 |
| **基础设施** | `config.py`（环境变量配置）、`logging.py`（JSON 结构化日志 + trace_id） |
| **数据模型** | `schemas.py`（Pydantic 请求/响应模型，严格对齐设计文档 §3 协议） |
| **服务层** | `vad_service.py`（流式 VAD + 动态阈值断句）、`asr_service.py`（异步 httpx 调 vLLM）、`itn_service.py`（线程池 ITN） |
| **API 层** | `websocket.py`（全链路处理）、`connection_manager.py`（并发控制）、`session.py`（会话状态机） |
| **运维** | `health.py`（健康探针）、`metrics.py`（Prometheus 指标）|
| **依赖** | 虚拟环境搭建 + 全部依赖安装（含 WeTextProcessing），服务本地启动验证通过 |

### 🔲 待完成

| 优先级 | 任务 | 说明 |
|:--|:--|:--|
| **P0** | 端到端联调 | 连接远程 vLLM ASR 模型，验证完整的 音频→VAD→ASR→ITN→推送 链路 |
| **P0** | VAD 流式断句验证 | 用真实音频测试 `StreamingVADSession` 的断句准确性和时间戳正确性 |
| **P1** | WebSocket 客户端测试脚本 | 编写模拟客户端，自动发送音频文件进行功能回归测试 |
| **P1** | docker-compose.yaml 编写 | 基于 `vllm-ascend:0.18` 镜像编排，配置 NPU 设备映射、Volume 挂载、环境变量注入 |
| **P1** | Dockerfile 编写 | 基础镜像打包（代码 + 依赖），权重通过 Volume 挂载 |
| **P2** | 异常恢复与重试 | vLLM 调用超时/失败时的重试策略、断线重连机制 |
| **P2** | 压力测试 | 多并发 WebSocket 连接压测，验证并发上限、内存稳定性、延迟分布 |
| **P2** | 单元测试 | 核心模块（VAD 动态阈值、音频编解码、Schema 序列化）的单元测试覆盖 |
| **P3** | Grafana 监控面板 | 基于 Prometheus 指标搭建可视化监控仪表盘 |

