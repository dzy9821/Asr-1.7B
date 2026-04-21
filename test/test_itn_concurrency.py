import asyncio
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.services.itn_service import ITNService
from src.core.config import settings

async def main():
    service = ITNService()
    service.startup()
    
    text = "十六个工作线程，一个itn实例。"
    
    # 预热一次
    await service.normalize(text)
    
    concurrency_levels = [1, 10, 20, 50, 100]
    
    print("=" * 60)
    print(f"ITN 多并发延迟测试 (工作线程数 ITN_WORKERS: {settings.ITN_WORKERS})")
    print("=" * 60)
    
    for c in concurrency_levels:
        start_time = time.time()
        
        # 创建并发任务
        tasks = [service.normalize(text) for _ in range(c)]
        
        # 并发执行
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time_ms = (end_time - start_time) * 1000
        avg_time_ms = total_time_ms / c
        
        print(f"并发请求数: {c:3d} | 总耗时: {total_time_ms:6.2f} ms | 平均单句耗时: {avg_time_ms:6.2f} ms")
        
        # 简单校验结果一致性
        assert all(r == results[0] for r in results), "结果不一致！"

    print("=" * 60)
    print("测试输出样本：")
    print(f"[{text}] -> [{results[0]}]")
    print("=" * 60)
    
    service.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
