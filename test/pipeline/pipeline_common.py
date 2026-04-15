"""
Common helpers for ASR pipelines.
"""
import base64
import io
import os
import re

import numpy as np


def clear_proxy_env():
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(key, None)


def load_audio(audio_path):
    import soundfile as sf

    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio, sr


def encode_audio_segment_to_data_url(audio_segment, sr):
    import soundfile as sf

    if not isinstance(audio_segment, np.ndarray):
        audio_segment = np.asarray(audio_segment, dtype=np.float32)
    if audio_segment.dtype != np.float32:
        audio_segment = audio_segment.astype(np.float32)

    buffer = io.BytesIO()
    sf.write(buffer, audio_segment, sr, format="WAV")
    audio_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:audio/wav;base64,{audio_base64}"


def create_asr_client(base_url, api_key="EMPTY"):
    import httpx
    from openai import OpenAI

    clear_proxy_env()
    http_client = httpx.Client()
    return OpenAI(base_url=base_url, api_key=api_key, http_client=http_client)


def parse_hotwords(hotwords):
    if hotwords is None:
        return []

    if isinstance(hotwords, str):
        parts = re.split(r"[,\n;，；]+", hotwords)
        return [part.strip() for part in parts if part.strip()]

    return [str(word).strip() for word in hotwords if str(word).strip()]


def build_asr_system_prompt(hotwords=None, system_prompt=None):
    lines = []
    if system_prompt and str(system_prompt).strip():
        lines.append(str(system_prompt).strip())

    hotword_list = parse_hotwords(hotwords)
    if hotword_list:
        hotword_text = "、".join(hotword_list)
        lines.append(f"请优先识别并准确输出以下热词：{hotword_text}。")

    return "\n".join(lines).strip()


def clean_asr_output(text):
    text = text if isinstance(text, str) else str(text)
    text = text.strip()

    if "<asr_text>" not in text:
        return text

    parts = re.split(r"(?:language\s+[^\s<]+)?<asr_text>", text, flags=re.IGNORECASE)
    cleaned = "".join(part.strip() for part in parts if part.strip())
    return cleaned.strip()


def asr_recognize(
    audio_segment,
    sr,
    model,
    base_url,
    api_key="EMPTY",
    client=None,
    hotwords=None,
    system_prompt=None,
):
    if client is None:
        client = create_asr_client(base_url=base_url, api_key=api_key)

    data_url = encode_audio_segment_to_data_url(audio_segment, sr)
    system_prompt_text = build_asr_system_prompt(
        hotwords=hotwords,
        system_prompt=system_prompt,
    )
    messages = []
    if system_prompt_text:
        messages.append({"role": "system", "content": system_prompt_text})
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {"url": data_url},
                }
            ],
        }
    )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    content_text = content if isinstance(content, str) else str(content)
    return clean_asr_output(content_text)


def vad_split_audio(audio, sr, vad_model):
    return vad_model.get_speech_segments(audio, sr)


def itn_normalize(text, itn_model):
    return itn_model.process(text)


def run_pipeline(
    audio_path,
    vad_model,
    itn_model,
    asr_model="Qwen3-ASR-1.7B",
    asr_base_url="http://148.148.52.127:15002/v1",
    asr_api_key="EMPTY",
    asr_hotwords=None,
    asr_system_prompt=None,
):
    audio, sr = load_audio(audio_path)
    segments = vad_split_audio(audio, sr, vad_model)
    client = create_asr_client(base_url=asr_base_url, api_key=asr_api_key)

    results = []
    for start, end in segments:
        if end <= start:
            continue
        segment = audio[start:end]
        asr_result = asr_recognize(
            segment,
            sr,
            model=asr_model,
            base_url=asr_base_url,
            api_key=asr_api_key,
            client=client,
            hotwords=asr_hotwords,
            system_prompt=asr_system_prompt,
        )
        if not asr_result:
            continue
        itn_result = itn_normalize(asr_result, itn_model)
        results.append(itn_result)

    return "".join(results)
