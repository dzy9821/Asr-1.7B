"""
ITN (逆文本正则化) 服务 —— 通过线程池调用 ITNProcessor。
"""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)

# 模块级 ITNProcessor 实例（延迟初始化）
_itn_processor = None


def _get_processor():
    """获取或创建 ITNProcessor 单例。"""
    global _itn_processor
    if _itn_processor is None:
        # 导入放在这里，避免在未安装 WeTextProcessing 时报错
        import sys

        models_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "models", "itn")
        )
        if models_dir not in sys.path:
            sys.path.insert(0, models_dir)

        from itn_wrapper import ITNProcessor  # noqa: E402

        _itn_processor = ITNProcessor()
        logger.info("ITN model loaded from %s", models_dir)
    return _itn_processor


def _run_itn(text: str) -> str:
    """在线程池中执行的同步 ITN 处理。"""
    processor = _get_processor()
    return processor.process(text)


class ITNService:
    """ITN 逆正则化服务，使用线程池避免阻塞事件循环。"""

    def __init__(self) -> None:
        self._pool: Optional[ThreadPoolExecutor] = None

    def startup(self) -> None:
        """初始化线程池。"""
        self._pool = ThreadPoolExecutor(
            max_workers=settings.ITN_WORKERS,
            thread_name_prefix="itn",
        )
        # 预热：在一个线程中加载模型
        future = self._pool.submit(_get_processor)
        future.result()
        logger.info("ITN service started with %d workers", settings.ITN_WORKERS)

    def shutdown(self) -> None:
        """关闭线程池。"""
        if self._pool:
            self._pool.shutdown(wait=False)
            self._pool = None

    async def normalize(self, text: str) -> str:
        """
        异步执行逆正则化。

        Args:
            text: ASR 原始输出文本

        Returns:
            逆正则化后的标准化文本
        """
        if not text or not text.strip():
            return ""
        assert self._pool is not None, "ITNService not started"
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool, _run_itn, text)
