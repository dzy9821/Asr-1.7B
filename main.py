"""
ASR 实时流式转录服务入口。

启动方式：
    python main.py
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.core.config import settings

# ---- 将 config.py 的 WS Ping 配置注入环境变量，供 uvicorn CLI 启动时自动读取 ----
# uvicorn 在 CLI 模式下通过 UVICORN_WS_PING_INTERVAL / UVICORN_WS_PING_TIMEOUT
# 环境变量读取 ping 参数。在模块加载时设置，确保无论哪种启动方式都生效。
os.environ.setdefault("UVICORN_WS_PING_INTERVAL", str(int(settings.WS_PING_INTERVAL)))
os.environ.setdefault("UVICORN_WS_PING_TIMEOUT", str(int(settings.WS_PING_TIMEOUT)))

from src.api.health import router as health_router
from src.api.metrics import router as metrics_router
from src.api.websocket import asr_service, itn_pool, router as ws_router
from src.core.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动/关闭资源。"""
    # ---- 启动 ----
    logger.info(
        "Starting ASR service on %s:%d (max_conn=%d, ping_interval=%.0f, ping_timeout=%.0f)",
        settings.WS_HOST,
        settings.WS_PORT,
        settings.MAX_CONNECTIONS,
        settings.WS_PING_INTERVAL,
        settings.WS_PING_TIMEOUT,
    )

    # TEN-VAD 为每连接独立实例，无需全局初始化

    # ITN 多进程池（eager init）
    itn_pool.start()
    logger.info("ITN pool ready: %d workers", itn_pool.num_workers)

    # ASR HTTP 客户端
    await asr_service.startup()

    logger.info("All services initialized")

    yield

    # ---- 关闭 ----
    logger.info("Shutting down ASR service...")
    await asr_service.shutdown()
    itn_pool.shutdown()
    logger.info("Shutdown complete")


app = FastAPI(
    title="ASR Real-time Streaming Service",
    version="0.1.0",
    lifespan=lifespan,
)

# 注册路由
app.include_router(ws_router)
app.include_router(health_router)
app.include_router(metrics_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.WS_HOST,
        port=settings.WS_PORT,
        log_level=settings.LOG_LEVEL.lower(),
        ws_ping_interval=settings.WS_PING_INTERVAL,
        ws_ping_timeout=settings.WS_PING_TIMEOUT,
    )
