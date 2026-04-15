"""
ASR pipeline with TEN VAD.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from models.itn.itn_wrapper import ITNProcessor
from models.vad.ten_vad_wrapper import TenVADWrapper

from pipeline_common import (
    asr_recognize,
    create_asr_client,
    encode_audio_segment_to_data_url,
    itn_normalize,
    load_audio,
    run_pipeline,
    vad_split_audio,
)


def _get_env_float(name, default=0.0):
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return float(value)


def main(
    audio_path,
    asr_model="Qwen3-ASR-1.7B",
    asr_base_url="http://148.148.52.127:15002/v1",
    asr_api_key="EMPTY",
    asr_context=None,
    vad_min_silence_duration_ms=None,
    asr_hotwords=None,
    asr_system_prompt=None,
):
    if asr_context is None:
        asr_context = os.getenv("ASR_CONTEXT")

    if asr_hotwords is None:
        asr_hotwords = os.getenv("ASR_HOTWORDS")
    if asr_system_prompt is None:
        asr_system_prompt = os.getenv("ASR_SYSTEM_PROMPT")

    if vad_min_silence_duration_ms is None:
        vad_min_silence_duration_ms = _get_env_float(
            "VAD_MIN_SILENCE_DURATION_MS", 0.0
        )

    vad_model = TenVADWrapper(
        min_silence_duration_ms=vad_min_silence_duration_ms,
    )
    itn_model = ITNProcessor()
    return run_pipeline(
        audio_path=audio_path,
        vad_model=vad_model,
        itn_model=itn_model,
        asr_model=asr_model,
        asr_base_url=asr_base_url,
        asr_api_key=asr_api_key,
        asr_context=asr_context,
        asr_hotwords=asr_hotwords,
        asr_system_prompt=asr_system_prompt,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python asr_pipeline_tenvad.py <audio_path>")
    print(main(sys.argv[1]))
