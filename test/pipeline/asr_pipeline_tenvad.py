"""
ASR pipeline with TEN VAD.
"""
import sys

import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from models.itn.itn_wrapper import ITNProcessor
from models.vad.ten_vad_wrapper import TenVADWrapper

from pipeline_common import (
    DEFAULT_ASR_CONTEXT,
    DEFAULT_VAD_MIN_SILENCE_DURATION_MS,
    asr_recognize,
    create_asr_client,
    encode_audio_segment_to_data_url,
    itn_normalize,
    load_audio,
    run_pipeline,
    vad_split_audio,
)

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
        asr_context = DEFAULT_ASR_CONTEXT

    if asr_hotwords is None:
        asr_hotwords = None
    if asr_system_prompt is None:
        asr_system_prompt = None

    if vad_min_silence_duration_ms is None:
        vad_min_silence_duration_ms = DEFAULT_VAD_MIN_SILENCE_DURATION_MS

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
