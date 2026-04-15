"""
ASR完整流程：VAD断点检测 -> ASR识别 -> ITN逆正则化 -> 结果拼接
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from models.vad.silero_vad_wrapper import SileroVAD
from models.itn.itn_wrapper import ITNProcessor


def load_audio(audio_path):
    """加载音频文件"""
    import soundfile as sf
    audio, sr = sf.read(audio_path)
    return audio, sr


def vad_split_audio(audio, sr, vad_model):
    """使用VAD模型分割音频"""
    print("[VAD] 开始断点检测...")
    segments = vad_model.get_speech_segments(audio, sr)
    print(f"[VAD] 检测到 {len(segments)} 个语音片段")
    return segments


def asr_recognize(audio_segment, sr):
    """调用ASR模型进行识别

    注意：当前环境无法调用ASR模型，此处为占位符
    实际使用时需要替换为真实的ASR调用
    """
    print(f"[ASR] 识别音频片段 (长度: {len(audio_segment)/sr:.2f}s)...")
    # TODO: 调用ASR模型
    # result = asr_model.recognize(audio_segment, sr)
    # return result
    return "这是一句话"  # 占位符


def itn_normalize(text, itn_model):
    """使用ITN模型进行逆正则化"""
    print(f"[ITN] 逆正则化: '{text}'")
    normalized = itn_model.process(text)
    print(f"[ITN] 结果: '{normalized}'")
    return normalized


def main(audio_path):
    """主流程"""
    print(f"开始处理音频: {audio_path}\n")

    # 1. 加载VAD模型
    vad_model = SileroVAD()

    # 2. 加载ITN模型
    itn_model = ITNProcessor()

    # 3. 加载音频
    audio, sr = load_audio(audio_path)
    print(f"[加载] 音频采样率: {sr}Hz, 时长: {len(audio)/sr:.2f}s\n")

    # 4. VAD分割
    segments = vad_split_audio(audio, sr, vad_model)

    # 5. 逐段识别和逆正则化
    results = []
    for i, (start, end) in enumerate(segments, 1):
        audio_segment = audio[start:end]

        # ASR识别
        asr_result = asr_recognize(audio_segment, sr)

        # ITN逆正则化
        itn_result = itn_normalize(asr_result, itn_model)
        results.append(itn_result)
        print()

    # 6. 拼接结果
    final_result = "".join(results)
    print("=" * 50)
    print(f"最终识别结果: {final_result}")
    print("=" * 50)

    return final_result


if __name__ == "__main__":
    # 示例音频路径
    audio_path = "/path/to/audio.wav"

    if len(sys.argv) > 1:
        audio_path = sys.argv[1]

    if not os.path.exists(audio_path):
        print(f"错误: 音频文件不存在 {audio_path}")
        sys.exit(1)

    main(audio_path)
