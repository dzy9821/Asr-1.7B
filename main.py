"""
ASR 实时流式转录服务入口。

启动方式：
    python -m uvicorn main:app --host 0.0.0.0 --port 8000 --ws-ping-interval 20 --ws-ping-timeout 300
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.health import router as health_router
from src.api.metrics import router as metrics_router
from src.api.websocket import asr_service, itn_pool, router as ws_router
from src.core.config import settings
from src.core.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动/关闭资源。"""
    # ---- 启动 ----
    logger.info(
        "Starting ASR service on %s:%d (max_conn=%d)",
        settings.WS_HOST,
        settings.WS_PORT,
        settings.MAX_CONNECTIONS,
    )

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
else:
    # 通过 CLI 启动时（uvicorn main:app），ping 超时由 CLI 参数或环境变量控制。
    # 若未显式指定，uvicorn 默认 ws_ping_timeout=20s，在高并发下极易触发。
    # 请使用：uvicorn main:app --ws-ping-interval 20 --ws-ping-timeout 300
    import os

    if not os.environ.get("UVICORN_WS_PING_TIMEOUT"):
        logger.warning(
            "⚠ Server started via CLI without --ws-ping-timeout. "
            "Default is 20s which may cause keepalive failures under load. "
            "Recommend: uvicorn main:app --ws-ping-interval 20 --ws-ping-timeout 300"
        )
