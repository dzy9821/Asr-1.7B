"""
模型测试脚本 - 验证VAD和ITN模型能否正常运行
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from models.vad.silero_vad_wrapper import SileroVAD
from models.itn.itn_wrapper import ITNProcessor


def test_vad_model():
    """测试VAD模型"""
    print("=" * 50)
    print("测试VAD模型")
    print("=" * 50)

    try:
        # 初始化VAD模型
        vad = SileroVAD()
        print("✓ VAD模型加载成功")

        # 生成测试音频（16kHz, 2秒）
        sr = 16000
        duration = 2
        audio = np.random.randn(sr * duration).astype(np.float32)
        print(f"✓ 生成测试音频: {duration}s, 采样率{sr}Hz")

        # 运行VAD
        segments = vad.get_speech_segments(audio, sr)
        print(f"✓ VAD检测完成，找到 {len(segments)} 个片段")

        if segments:
            for i, (start, end) in enumerate(segments, 1):
                duration_seg = (end - start) / sr
                print(f"  片段{i}: {start/sr:.2f}s - {end/sr:.2f}s (时长: {duration_seg:.2f}s)")

        print("✓ VAD模型测试通过\n")
        return True

    except Exception as e:
        print(f"✗ VAD模型测试失败: {e}\n")
        import traceback

        traceback.print_exc()
        return False


def test_itn_model():
    """测试ITN模型"""
    print("=" * 50)
    print("测试ITN模型")
    print("=" * 50)

    try:
        # 初始化ITN模型
        itn = ITNProcessor()
        print("✓ ITN模型加载成功")

        # 测试文本
        test_texts = [
            "今天天气很好",
            "一二三四五",
            "百分之九十五",
        ]

        for text in test_texts:
            result = itn.process(text)
            print(f"✓ 输入: '{text}' -> 输出: '{result}'")

        print("✓ ITN模型测试通过\n")
        return True

    except Exception as e:
        print(f"✗ ITN模型测试失败: {e}\n")
        import traceback

        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n开始模型测试...\n")

    vad_ok = test_vad_model()
    itn_ok = test_itn_model()

    print("=" * 50)
    print("测试总结")
    print("=" * 50)
    print(f"VAD模型: {'✓ 通过' if vad_ok else '✗ 失败'}")
    print(f"ITN模型: {'✓ 通过' if itn_ok else '✗ 失败'}")

    if vad_ok and itn_ok:
        print("\n✓ 所有模型测试通过")
        return 0
    else:
        print("\n✗ 部分模型测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
