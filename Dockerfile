# ---- 构建 ----
FROM quay.io/ascend/vllm-ascend:v0.18.0rc1-310p

# ---- 代理（构建时按需传入 --build-arg HTTP_PROXY=...） ----
ARG HTTP_PROXY
ENV http_proxy=${HTTP_PROXY}
ENV https_proxy=${HTTP_PROXY}
ENV no_proxy=localhost,127.0.0.1

# ---- 1. 系统依赖（编译 OpenFst） ----
RUN apt-get update && \
    apt-get install -y --no-install-recommends g++ make wget curl && \
    rm -rf /var/lib/apt/lists/*

# ---- 2. 编译安装 OpenFst ----
RUN cd /tmp && \
    wget --no-check-certificate https://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.8.3.tar.gz && \
    tar xzf openfst-1.8.3.tar.gz && \
    cd openfst-1.8.3 && \
    ./configure --prefix=/usr/local --enable-grm --enable-static --enable-shared && \
    make -j$(nproc) && make install && \
    cd / && rm -rf /tmp/openfst-1.8.3*

# ---- 3. 注册 OpenFst 库路径 ----
RUN echo /usr/local/lib >> /etc/ld.so.conf.d/openfst.conf && ldconfig
ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}

# ---- 4. 安装 pynini + WeTextProcessing ----
RUN pip install pynini==2.1.6 && \
    GIT_SSL_NO_VERIFY=1 pip install 'git+https://github.com/wenet-e2e/WeTextProcessing.git'

# ---- 5. 安装项目 Python 依赖（torch/vllm 已内置，不要重装以免破坏兼容） ----
RUN pip install \
    "librosa" \
    "torchaudio>=2.0.0" \
    "fastapi>=0.115.0" \
    "websockets>=12.0" \
    "uvicorn[standard]>=0.30.0" \
    "pydantic>=2.5.0" \
    "numpy==1.26.4" \
    "httpx>=0.27.0" \
    "prometheus-client>=0.21.0" \
    "soundfile>=0.12.0"

# ---- 6. 运行时系统库（放在 pip install 之后保留缓存） ----
RUN apt-get update && \
    apt-get install -y --no-install-recommends libopus0 libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# ---- 7. 复制项目 ----
WORKDIR /app
COPY . .

# ---- 7. ONNX 模型软链接（VAD 需要的相对路径） ----
RUN ln -sf models/vad/ten-vad/onnx_model onnx_model

# ---- 8. VAD 原生库路径 ----
ENV LD_LIBRARY_PATH=/app/models/vad/ten-vad/lib/Linux/aarch64:/usr/local/lib:${LD_LIBRARY_PATH}

# ---- 9. vLLM 默认配置（容器内回环访问） ----
ENV VLLM_API_BASE=http://127.0.0.1:15002/v1

# ---- 10. 清除构建代理（避免泄露到运行时） ----
ENV http_proxy="" https_proxy="" no_proxy=""

# ---- 11. 端口（仅声明，实际映射在 docker-compose 中配置） ----
EXPOSE 8000 15002

# ---- 12. 启动 ----
CMD ["python", "main.py"]
