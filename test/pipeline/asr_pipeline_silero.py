"""
ASR pipeline with Silero VAD.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from models.itn.itn_wrapper import ITNProcessor
from models.vad.silero_vad_wrapper import SileroVAD

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
    vad_model = SileroVAD()
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
        raise SystemExit("Usage: python asr_pipeline_silero.py <audio_path>")
    print(main(sys.argv[1]))

