"""
测试：一个 TenVad 实例能否同时服务两个音频流。

实验设计：
  实验1 - 状态污染检测：
    - 将同一段音频分成两份模拟两个连接
    - 对比"独占实例顺序处理"与"共享实例交替处理"的 flag 差异
    - 如果 flag 不一致 → 内部隐藏状态被污染，共享不可行

  实验2 - 逐帧丢弃上下文：
    - 每帧 create → process → destroy，对比持久实例的结果
    - 测量准确率差异和性能开销

  实验3 - 并发安全性（线程）：
    - 同一个 TenVad 实例在两个线程中同时 process
    - 检测是否会崩溃或结果错乱

用法：
    cd /home/ubuntu/project/ASR-1.7B/Asr-1.7B
    python -m test.test_vad_shared_instance
"""

from __future__ import annotations

import os
import sys
import time
import threading
from pathlib import Path

import numpy as np

# ---------- 路径设置 ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VAD_DIR = PROJECT_ROOT / "models" / "vad"
INCLUDE_DIR = VAD_DIR / "ten-vad" / "include"

sys.path.insert(0, str(INCLUDE_DIR))
from ten_vad import TenVad  # noqa: E402

AUDIO_PATH = PROJECT_ROOT / "120报警电话16k.wav"

HOP_SIZE = 640
THRESHOLD = 0.5
SAMPLE_RATE = 16000


# ============================================================
# 工具函数
# ============================================================

def load_wav_pcm16(path: str | Path) -> np.ndarray:
    """加载 wav 文件，返回 int16 PCM 数组。"""
    import wave
    with wave.open(str(path), "rb") as wf:
        assert wf.getnchannels() == 1, f"Expected mono, got {wf.getnchannels()} channels"
        assert wf.getsampwidth() == 2, f"Expected 16-bit, got {wf.getsampwidth() * 8}-bit"
        assert wf.getframerate() == SAMPLE_RATE, f"Expected {SAMPLE_RATE}Hz, got {wf.getframerate()}Hz"
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    return pcm


def split_frames(pcm: np.ndarray, hop_size: int) -> list[np.ndarray]:
    """将 PCM 数组切成定长帧列表。"""
    n_frames = len(pcm) // hop_size
    frames = [pcm[i * hop_size : (i + 1) * hop_size] for i in range(n_frames)]
    return frames


def process_frames_with_instance(vad: TenVad, frames: list[np.ndarray]) -> list[tuple[float, int]]:
    """用一个已有实例逐帧处理，返回 (probability, flag) 列表。"""
    results = []
    for frame in frames:
        prob, flag = vad.process(frame)
        results.append((prob, flag))
    return results


def process_frames_fresh_per_frame(frames: list[np.ndarray]) -> list[tuple[float, int]]:
    """每帧创建新实例 → process → destroy（模拟丢弃上下文）。"""
    results = []
    for frame in frames:
        vad = TenVad(hop_size=HOP_SIZE, threshold=THRESHOLD)
        prob, flag = vad.process(frame)
        results.append((prob, flag))
        del vad  # 触发 __del__ → ten_vad_destroy
    return results


# ============================================================
# 实验1：状态污染检测
# ============================================================

def experiment_1_state_corruption():
    """
    对比独占 vs 共享实例的输出差异。
    
    - 独占：实例A处理流A全部帧；实例B处理流B全部帧
    - 共享：单一实例交替处理流A帧和流B帧
    """
    print("=" * 70)
    print("实验1：状态污染检测 (独占 vs 共享交替)")
    print("=" * 70)

    pcm = load_wav_pcm16(AUDIO_PATH)
    all_frames = split_frames(pcm, HOP_SIZE)
    n = len(all_frames)
    mid = n // 2

    # 将音频分成两个"流"
    stream_a_frames = all_frames[:mid]
    stream_b_frames = all_frames[mid:]
    print(f"音频总帧数: {n}, 流A: {len(stream_a_frames)}, 流B: {len(stream_b_frames)}")

    # ---- 独占模式 ----
    vad_a = TenVad(hop_size=HOP_SIZE, threshold=THRESHOLD)
    vad_b = TenVad(hop_size=HOP_SIZE, threshold=THRESHOLD)
    exclusive_a = process_frames_with_instance(vad_a, stream_a_frames)
    exclusive_b = process_frames_with_instance(vad_b, stream_b_frames)
    del vad_a, vad_b

    # ---- 共享交替模式 ----
    vad_shared = TenVad(hop_size=HOP_SIZE, threshold=THRESHOLD)
    shared_a = []
    shared_b = []
    max_len = max(len(stream_a_frames), len(stream_b_frames))
    for i in range(max_len):
        if i < len(stream_a_frames):
            prob, flag = vad_shared.process(stream_a_frames[i])
            shared_a.append((prob, flag))
        if i < len(stream_b_frames):
            prob, flag = vad_shared.process(stream_b_frames[i])
            shared_b.append((prob, flag))
    del vad_shared

    # ---- 对比 ----
    diff_a = 0
    for i, (exc, sha) in enumerate(zip(exclusive_a, shared_a)):
        if exc[1] != sha[1]:
            diff_a += 1
    
    diff_b = 0
    for i, (exc, sha) in enumerate(zip(exclusive_b, shared_b)):
        if exc[1] != sha[1]:
            diff_b += 1

    print(f"\n流A flag差异帧数: {diff_a}/{len(exclusive_a)} ({diff_a/len(exclusive_a)*100:.1f}%)")
    print(f"流B flag差异帧数: {diff_b}/{len(exclusive_b)} ({diff_b/len(exclusive_b)*100:.1f}%)")
    
    if diff_a > 0 or diff_b > 0:
        print("⚠ 结论：共享实例交替处理会导致 flag 结果不一致 → 内部状态被污染！")
        print("  一个 TenVad 实例不能同时服务两个音频流。")
        
        # 打印前20个差异详情
        print("\n  流A 前20个差异帧详情:")
        count = 0
        for i, (exc, sha) in enumerate(zip(exclusive_a, shared_a)):
            if exc[1] != sha[1]:
                t = i * HOP_SIZE / SAMPLE_RATE
                print(f"    帧{i:4d} ({t:6.2f}s): 独占flag={exc[1]} prob={exc[0]:.3f} | 共享flag={sha[1]} prob={sha[0]:.3f}")
                count += 1
                if count >= 20:
                    break
    else:
        print("✓ 结论：共享实例交替处理的 flag 与独占一致（需进一步验证 probability 精度）")
        # 检查 probability 差异
        prob_diffs_a = [abs(exc[0] - sha[0]) for exc, sha in zip(exclusive_a, shared_a)]
        max_prob_diff = max(prob_diffs_a) if prob_diffs_a else 0
        avg_prob_diff = sum(prob_diffs_a) / len(prob_diffs_a) if prob_diffs_a else 0
        print(f"  流A probability最大差异: {max_prob_diff:.6f}, 平均差异: {avg_prob_diff:.6f}")

    return diff_a > 0 or diff_b > 0


# ============================================================
# 实验2：逐帧丢弃上下文 vs 持久实例
# ============================================================

def experiment_2_per_frame_create():
    """
    对比持久实例 vs 每帧新建实例的准确率和性能。
    """
    print("\n" + "=" * 70)
    print("实验2：逐帧丢弃上下文 (per-frame create/destroy)")
    print("=" * 70)

    pcm = load_wav_pcm16(AUDIO_PATH)
    frames = split_frames(pcm, HOP_SIZE)
    # 取前500帧测试（避免太慢）
    test_frames = frames[:500]
    print(f"测试帧数: {len(test_frames)} (总帧数: {len(frames)})")

    # ---- 持久实例 (baseline) ----
    vad = TenVad(hop_size=HOP_SIZE, threshold=THRESHOLD)
    t0 = time.perf_counter()
    baseline = process_frames_with_instance(vad, test_frames)
    t_baseline = time.perf_counter() - t0
    del vad

    # ---- 每帧新建实例 ----
    t0 = time.perf_counter()
    per_frame = process_frames_fresh_per_frame(test_frames)
    t_per_frame = time.perf_counter() - t0

    # ---- 对比 ----
    flag_diff = sum(1 for b, p in zip(baseline, per_frame) if b[1] != p[1])
    prob_diffs = [abs(b[0] - p[0]) for b, p in zip(baseline, per_frame)]
    
    baseline_speech_frames = sum(1 for b in baseline if b[1] == 1)
    per_frame_speech_frames = sum(1 for p in per_frame if p[1] == 1)

    print(f"\n持久实例耗时: {t_baseline*1000:.1f}ms  ({t_baseline/len(test_frames)*1000:.3f}ms/帧)")
    print(f"逐帧新建耗时: {t_per_frame*1000:.1f}ms  ({t_per_frame/len(test_frames)*1000:.3f}ms/帧)")
    print(f"性能倍数: {t_per_frame/t_baseline:.1f}x")
    print(f"\nflag 差异帧数: {flag_diff}/{len(test_frames)} ({flag_diff/len(test_frames)*100:.1f}%)")
    print(f"持久实例语音帧: {baseline_speech_frames}, 逐帧新建语音帧: {per_frame_speech_frames}")
    print(f"probability 最大差异: {max(prob_diffs):.6f}, 平均差异: {sum(prob_diffs)/len(prob_diffs):.6f}")

    if flag_diff > 0:
        print("\n⚠ 逐帧丢弃上下文导致 flag 结果不同！")
        print("  前20个差异帧:")
        count = 0
        for i, (b, p) in enumerate(zip(baseline, per_frame)):
            if b[1] != p[1]:
                t = i * HOP_SIZE / SAMPLE_RATE
                print(f"    帧{i:4d} ({t:6.2f}s): 持久flag={b[1]} prob={b[0]:.3f} | 逐帧flag={p[1]} prob={p[0]:.3f}")
                count += 1
                if count >= 20:
                    break
    else:
        print("✓ 逐帧丢弃上下文的 flag 结果与持久实例一致")

    return flag_diff, t_baseline, t_per_frame


# ============================================================
# 实验3：线程安全性测试
# ============================================================

def experiment_3_thread_safety():
    """
    测试同一个 TenVad 实例在两个线程中同时 process 是否安全。
    """
    print("\n" + "=" * 70)
    print("实验3：线程安全性测试 (同一实例, 双线程)")
    print("=" * 70)

    pcm = load_wav_pcm16(AUDIO_PATH)
    frames = split_frames(pcm, HOP_SIZE)
    mid = len(frames) // 2
    stream_a = frames[:mid]
    stream_b = frames[mid:]

    vad_shared = TenVad(hop_size=HOP_SIZE, threshold=THRESHOLD)
    
    results_a = []
    results_b = []
    errors = []

    def worker(stream, result_list, name):
        try:
            for frame in stream:
                prob, flag = vad_shared.process(frame)
                result_list.append((prob, flag))
        except Exception as e:
            errors.append((name, e))

    t1 = threading.Thread(target=worker, args=(stream_a, results_a, "A"))
    t2 = threading.Thread(target=worker, args=(stream_b, results_b, "B"))

    t0 = time.perf_counter()
    t1.start()
    t2.start()
    t1.join(timeout=30)
    t2.join(timeout=30)
    elapsed = time.perf_counter() - t0

    del vad_shared

    if errors:
        print(f"⚠ 线程执行出错: {errors}")
    else:
        print(f"双线程执行完成, 耗时: {elapsed*1000:.1f}ms")
        print(f"线程A 处理帧数: {len(results_a)}/{len(stream_a)}")
        print(f"线程B 处理帧数: {len(results_b)}/{len(stream_b)}")

        # 对比独占模式
        vad_solo_a = TenVad(hop_size=HOP_SIZE, threshold=THRESHOLD)
        solo_a = process_frames_with_instance(vad_solo_a, stream_a)
        del vad_solo_a

        diff = sum(1 for s, r in zip(solo_a, results_a) if s[1] != r[1])
        print(f"线程A vs 独占A flag差异: {diff}/{min(len(solo_a), len(results_a))}")

        if diff > 0:
            print("⚠ 并发访问导致结果不一致 → TenVad 非线程安全！")
        else:
            print("✓ 并发结果一致（但可能有 GIL 保护，不代表真正线程安全）")


# ============================================================
# 实验4：两个流串行复用同一实例（每个流处理前 reset 的可行性）
# ============================================================

def experiment_4_serial_reuse():
    """
    测试同一个实例先处理流A，再处理流B，流B的结果是否被流A的上下文污染。
    对比：独立实例处理流B 的结果。
    """
    print("\n" + "=" * 70)
    print("实验4：串行复用 (流A处理完 → 同实例继续处理流B)")
    print("=" * 70)

    pcm = load_wav_pcm16(AUDIO_PATH)
    frames = split_frames(pcm, HOP_SIZE)
    mid = len(frames) // 2
    stream_a = frames[:mid]
    stream_b = frames[mid:]

    # 独占实例处理流B (baseline)
    vad_solo = TenVad(hop_size=HOP_SIZE, threshold=THRESHOLD)
    solo_b = process_frames_with_instance(vad_solo, stream_b)
    del vad_solo

    # 复用实例：先处理流A，再处理流B（不重建实例）
    vad_reuse = TenVad(hop_size=HOP_SIZE, threshold=THRESHOLD)
    _ = process_frames_with_instance(vad_reuse, stream_a)  # 先跑流A
    reuse_b = process_frames_with_instance(vad_reuse, stream_b)  # 再跑流B
    del vad_reuse

    diff = sum(1 for s, r in zip(solo_b, reuse_b) if s[1] != r[1])
    prob_diffs = [abs(s[0] - r[0]) for s, r in zip(solo_b, reuse_b)]
    
    print(f"流B flag差异帧数: {diff}/{len(solo_b)} ({diff/len(solo_b)*100:.1f}%)")
    if prob_diffs:
        print(f"probability 最大差异: {max(prob_diffs):.6f}, 平均差异: {sum(prob_diffs)/len(prob_diffs):.6f}")

    if diff > 0:
        print("⚠ 串行复用时，前一个流的上下文会影响后续流的结果 → 需要 reset/重建实例")
        print("  前10个差异帧:")
        count = 0
        for i, (s, r) in enumerate(zip(solo_b, reuse_b)):
            if s[1] != r[1]:
                t = i * HOP_SIZE / SAMPLE_RATE
                print(f"    帧{i:4d} ({t:6.2f}s): 独占flag={s[1]} prob={s[0]:.3f} | 复用flag={r[1]} prob={r[0]:.3f}")
                count += 1
                if count >= 10:
                    break
    else:
        print("✓ 串行复用结果一致（上下文可能是无状态的或自动衰减的）")


# ============================================================
# 主入口
# ============================================================

def main():
    print(f"音频文件: {AUDIO_PATH}")
    print(f"HOP_SIZE: {HOP_SIZE}, THRESHOLD: {THRESHOLD}")
    
    if not AUDIO_PATH.exists():
        print(f"ERROR: 音频文件不存在: {AUDIO_PATH}")
        sys.exit(1)

    # 实验1：核心可行性判定
    is_corrupted = experiment_1_state_corruption()

    # 实验2：备选方案 - 逐帧丢弃上下文
    experiment_2_per_frame_create()

    # 实验3：线程安全性
    experiment_3_thread_safety()

    # 实验4：串行复用（上下文残留）
    experiment_4_serial_reuse()

    # ---- 总结 ----
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    if is_corrupted:
        print("❌ 一个 TenVad 实例不能交替服务两个音频流（内部状态会互相污染）")
        print("   → 必须为每个连接维护独立的 TenVad 实例")
        print("   → 当前架构（每连接一个 StreamingVADSession）是正确的设计")
    else:
        print("✓ TenVad 可能是无状态的，可以探索共享实例方案")


if __name__ == "__main__":
    main()
