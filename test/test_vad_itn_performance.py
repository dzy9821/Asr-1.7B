import time
import os
import sys
import wave
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# 保证 src 模块能在 test 目录下正常引入
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.config import settings
from src.services.vad_service import StreamingVADSession
from src.services.itn_service import ITNService

def load_wav(file_path):
    """读取真实的 wav 音频文件"""
    with wave.open(file_path, 'rb') as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        audio_data = wf.readframes(n_frames)
        # 转换为 numpy int16 数组
        pcm_int16 = np.frombuffer(audio_data, dtype=np.int16)
        return pcm_int16, sample_rate

def process_stream(vad_session, audio_data, sample_rate=16000, chunk_size=640):
    """模拟流式输入并记录分段的时间戳"""
    timestamps = []
    
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i+chunk_size]
        if len(chunk) < chunk_size:
            break
        res = vad_session.feed_audio(chunk)
        if res:
            for seg in res:
                start_time = seg['start_sample'] / sample_rate
                end_time = seg['end_sample'] / sample_rate
                timestamps.append((round(start_time, 3), round(end_time, 3)))
                
    flush_res = vad_session.flush()
    if flush_res:
        start_time = flush_res['start_sample'] / sample_rate
        end_time = flush_res['end_sample'] / sample_rate
        timestamps.append((round(start_time, 3), round(end_time, 3)))
        
    return timestamps

# 1、测试两个连接共用一个vad的分段效果，输入分段的时间戳
def test_shared_vad(audio_data, sample_rate):
    print("\n--- 1. 测试两个连接共用一个 VAD 的分段效果 ---")
    shared_vad = StreamingVADSession(sample_rate=sample_rate)
    
    # 模拟两个并发流同时使用共享的 VAD 实例
    with ThreadPoolExecutor(max_workers=2) as executor:
        future1 = executor.submit(process_stream, shared_vad, audio_data, sample_rate)
        future2 = executor.submit(process_stream, shared_vad, audio_data, sample_rate)
        
        timestamps1 = future1.result()
        timestamps2 = future2.result()
        
    print(f"【连接 1】获得的分段时间戳 (秒): {timestamps1}")
    print(f"【连接 2】获得的分段时间戳 (秒): {timestamps2}")

# 2、测试单个连接用vad的分段效果，输出分段的时间戳
def test_single_vad(audio_data, sample_rate):
    print("\n--- 2. 测试单个连接用自己的 VAD 的分段效果 ---")
    vad = StreamingVADSession(sample_rate=sample_rate)
    
    timestamps = process_stream(vad, audio_data, sample_rate)
    print(f"【正常独占】获得的分段时间戳 (秒): {timestamps}")

# 3、测试vad的加载速度，输出毫秒
def test_vad_loading_time():
    print("\n--- 3. 测试 VAD 的加载速度 ---")
    start = time.perf_counter()
    vad = StreamingVADSession()
    elapsed = (time.perf_counter() - start) * 1000  # 转换为毫秒
    print(f"单次实例化 VAD (加载耗时): {elapsed:.2f} ms")

# ITN 并发任务辅助函数
async def simulate_itn_requests(itn_service, requests_count, text):
    start = time.perf_counter()
    
    async def single_request():
        req_start = time.perf_counter()
        await itn_service.normalize(text)
        req_end = time.perf_counter()
        return req_end - req_start
        
    tasks = [single_request() for _ in range(requests_count)]
    times = await asyncio.gather(*tasks)
    
    total_time = time.perf_counter() - start
    avg_time = (sum(times) / requests_count) * 1000  # 毫秒
    return total_time, avg_time

# 4、输出100个并发时，使用单个itn实例，计算处理平均时长，注意：等待时间也算处理时间
def test_single_itn_100_concurrent():
    print("\n--- 4. 单个 ITN 实例，处理 100 个并发的平均时长 ---")
    # 临时覆盖全局设定，确保只有 1 个 ITN 工作线程(即单个 ITN 实例运算)
    original_workers = settings.ITN_WORKERS
    settings.ITN_WORKERS = 1
    
    itn_service = ITNService()
    itn_service.startup()
    
    text = "由于120报警电话属于紧急呼救，请您保持冷静并详细说明情况。"
    
    # 运行异步并发测试
    total_time, avg_time = asyncio.run(simulate_itn_requests(itn_service, 100, text))
    
    print(f"使用 ITN Worker 数: {settings.ITN_WORKERS}")
    print(f"100 并发总耗时: {total_time:.4f} 秒")
    print(f"【单个 ITN】每个并发请求的平均处理时长 (含排队等待时间): {avg_time:.2f} 毫秒")
    
    itn_service.shutdown()
    settings.ITN_WORKERS = original_workers # 恢复设置

# 5 的多进程执行函数
def _multiprocess_itn_task(text, count):
    # 子进程里强制设定 worker 数为 1，确保每个进程只有一个实例
    settings.ITN_WORKERS = 1
    
    itn_service = ITNService()
    itn_service.startup()
    
    async def run_batch():
        async def single_request():
            req_start = time.perf_counter()
            await itn_service.normalize(text)
            req_end = time.perf_counter()
            return req_end - req_start
            
        tasks = [single_request() for _ in range(count)]
        return await asyncio.gather(*tasks)
        
    times = asyncio.run(run_batch())
    itn_service.shutdown()
    return times

# 5、计算使用多进程创建4个itn实例，处理100个并发，每个并发的平均处理时间
def test_multiprocess_itn():
    print("\n--- 5. 使用多进程创建 4 个 ITN 实例，处理 100 个并发 ---")
    text = "由于120报警电话属于紧急呼救，请您保持冷静并详细说明情况。"
    num_processes = 4
    total_requests = 100
    requests_per_process = total_requests // num_processes
    
    start = time.perf_counter()
    
    # 使用 spawn 可以避免 Linux fork 时 C++ 模型内存状态混乱的问题
    ctx = multiprocessing.get_context("spawn")
    
    with ctx.Pool(processes=num_processes) as pool:
        # 启动 4 个进程，每个进程分配 25 个并发任务
        results = [
            pool.apply_async(_multiprocess_itn_task, args=(text, requests_per_process)) 
            for _ in range(num_processes)
        ]
        
        # 等待所有进程结束并获取各进程的每个请求执行时长
        all_times = []
        for res in results:
            all_times.extend(res.get())
        
    # 计算整体从发起进程池到全部完成的总时间
    total_time = time.perf_counter() - start
    avg_time = (sum(all_times) / total_requests) * 1000  # 毫秒
    
    print(f"开启进程数: {num_processes}，每进程处理数: {requests_per_process}")
    print(f"多进程总处理时间: {total_time:.4f} 秒")
    print(f"【多进程 4 个 ITN】每个并发请求的平均处理时长 (含排队等待时间): {avg_time:.2f} 毫秒")


if __name__ == "__main__":
    audio_path = "/home/d1465299/funasr-nano/data/120报警电话16k.wav"
    
    print(f"正在读取真实测试音频: {audio_path}")
    try:
        audio_data, sample_rate = load_wav(audio_path)
    except Exception as e:
        print(f"【严重错误】无法读取音频文件 {audio_path}，详情: {e}")
        sys.exit(1)
        
    # test_shared_vad(audio_data, sample_rate) # <--- 已注释，因为C++底层崩溃会导致程序直接终止
    test_single_vad(audio_data, sample_rate)
    test_vad_loading_time()
    test_single_itn_100_concurrent()
    test_multiprocess_itn()
