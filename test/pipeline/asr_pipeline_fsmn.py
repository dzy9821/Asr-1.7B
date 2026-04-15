"""
ASR pipeline with FSMN VAD.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from models.itn.itn_wrapper import ITNProcessor
from models.vad.fsmn_vad_wrapper import FSMNVADWrapper

from pipeline_common import (
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
):
    vad_model = FSMNVADWrapper()
    itn_model = ITNProcessor()
    return run_pipeline(
        audio_path=audio_path,
        vad_model=vad_model,
        itn_model=itn_model,
        asr_model=asr_model,
        asr_base_url=asr_base_url,
        asr_api_key=asr_api_key,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python asr_pipeline_fsmn.py <audio_path>")
    print(main(sys.argv[1]))

