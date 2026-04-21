"""
统一配置管理 —— 所有字段均可通过同名环境变量覆盖。
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """服务全局配置，基于 pydantic-settings 从环境变量读取。"""

    # ---- 服务参数 ----
    WS_HOST: str = "0.0.0.0"
    WS_PORT: int = 8000
    MAX_CONNECTIONS: int = 50
    HANDSHAKE_TIMEOUT: int = 5  # 握手超时（秒）

    # ---- 进程池 ----
    VAD_WORKERS: int = 16
    ITN_WORKERS: int = 16

    # ---- vLLM ----
    VLLM_API_BASE: str = "http://148.148.52.127:15002/v1"
    VLLM_MODEL_NAME: str = "Qwen3-ASR-1.7B"
    VLLM_API_KEY: str = "EMPTY"

    # ---- NPU ----
    ASCEND_RT_VISIBLE_DEVICES: str = "0"

    # ---- 日志 ----
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


# 全局单例
settings = Settings()
