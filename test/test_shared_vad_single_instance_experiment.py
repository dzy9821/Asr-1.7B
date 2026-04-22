import argparse
import json
import os
import subprocess
import sys
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DEFAULT_AUDIO_PATH = (
    "/home/ubuntu/project/ASR-1.7B/Asr-1.7B/120报警电话16k.wav"
)


def load_wav(file_path: str) -> tuple[np.ndarray, int]:
    with wave.open(file_path, "rb") as wf:
        if wf.getnchannels() != 1:
            raise ValueError(f"Only mono wav is supported, got {wf.getnchannels()} channels")
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        audio_data = wf.readframes(n_frames)
    return np.frombuffer(audio_data, dtype=np.int16), sample_rate


def _to_timestamps(segments: list[dict], sample_rate: int) -> list[tuple[float, float]]:
    timestamps: list[tuple[float, float]] = []
    for seg in segments:
        start_time = round(seg["start_sample"] / sample_rate, 3)
        end_time = round(seg["end_sample"] / sample_rate, 3)
        timestamps.append((start_time, end_time))
    return timestamps


def process_stream(
    conn_name: str,
    vad_session,
    audio_data: np.ndarray,
    sample_rate: int,
    chunk_size: int,
    start_barrier: threading.Barrier | None = None,
) -> dict:
    if start_barrier is not None:
        start_barrier.wait()

    timestamps: list[tuple[float, float]] = []
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i : i + chunk_size]
        if len(chunk) < chunk_size:
            break
        res = vad_session.feed_audio(chunk)
        if res:
            timestamps.extend(_to_timestamps(res, sample_rate))

        # 主动让两个线程更容易交替执行，放大共享状态问题。
        if (i // chunk_size) % 8 == 0:
            time.sleep(0)

    flush_res = vad_session.flush()
    if flush_res:
        timestamps.extend(_to_timestamps([flush_res], sample_rate))

    return {
        "connection": conn_name,
        "segment_count": len(timestamps),
        "timestamps": timestamps,
    }


def run_shared_vad_once(audio_data: np.ndarray, sample_rate: int, chunk_size: int) -> dict:
    from src.services.vad_service import StreamingVADSession

    shared_vad = StreamingVADSession(sample_rate=sample_rate, hop_size=chunk_size)
    barrier = threading.Barrier(2)

    with ThreadPoolExecutor(max_workers=2) as executor:
        future1 = executor.submit(
            process_stream,
            "conn-1",
            shared_vad,
            audio_data,
            sample_rate,
            chunk_size,
            barrier,
        )
        future2 = executor.submit(
            process_stream,
            "conn-2",
            shared_vad,
            audio_data,
            sample_rate,
            chunk_size,
            barrier,
        )
        conn1 = future1.result()
        conn2 = future2.result()

    return {"conn1": conn1, "conn2": conn2}


def _summarize_conn(result: dict) -> dict:
    timestamps = result["timestamps"]
    return {
        "connection": result["connection"],
        "segment_count": result["segment_count"],
        "first_5_timestamps": timestamps[:5],
        "last_5_timestamps": timestamps[-5:] if len(timestamps) > 5 else timestamps,
    }


def child_run(audio_path: str, rounds: int, chunk_size: int) -> int:
    try:
        audio_data, sample_rate = load_wav(audio_path)
    except Exception as exc:
        print(f"[child] failed to load wav: {exc}", file=sys.stderr)
        return 2

    all_rounds: list[dict] = []
    for idx in range(rounds):
        started = time.perf_counter()
        result = run_shared_vad_once(audio_data, sample_rate, chunk_size)
        elapsed_ms = (time.perf_counter() - started) * 1000
        all_rounds.append(
            {
                "round": idx + 1,
                "elapsed_ms": round(elapsed_ms, 2),
                "conn1": _summarize_conn(result["conn1"]),
                "conn2": _summarize_conn(result["conn2"]),
            }
        )

    output = {
        "audio_path": audio_path,
        "sample_rate": sample_rate,
        "chunk_size": chunk_size,
        "rounds": all_rounds,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def run_experiment_subprocess(
    audio_path: str, rounds: int, chunk_size: int, timeout_sec: int
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        __file__,
        "--child-run",
        "--audio",
        audio_path,
        "--rounds",
        str(rounds),
        "--chunk-size",
        str(chunk_size),
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)


def _format_return_code(return_code: int) -> str:
    if return_code < 0:
        return f"terminated by signal {-return_code}"
    return f"exit code {return_code}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experiment: two connections concurrently feed the same wav to one shared VAD instance."
    )
    parser.add_argument("--audio", default=DEFAULT_AUDIO_PATH)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=640)
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--child-run", action="store_true")
    args = parser.parse_args()

    if args.child_run:
        return child_run(args.audio, args.rounds, args.chunk_size)

    completed = run_experiment_subprocess(
        audio_path=args.audio,
        rounds=args.rounds,
        chunk_size=args.chunk_size,
        timeout_sec=args.timeout_sec,
    )
    print("=== shared VAD experiment ===")
    print(f"audio: {args.audio}")
    print(f"subprocess status: {_format_return_code(completed.returncode)}")
    if completed.stdout:
        print("----- child stdout -----")
        print(completed.stdout)
    if completed.stderr:
        print("----- child stderr -----", file=sys.stderr)
        print(completed.stderr, file=sys.stderr)

    return 0 if completed.returncode == 0 else completed.returncode


def test_shared_vad_single_instance_manual():
    import pytest

    if os.getenv("RUN_SHARED_VAD_EXPERIMENT") != "1":
        pytest.skip("manual experiment; set RUN_SHARED_VAD_EXPERIMENT=1 to run")

    if not Path(DEFAULT_AUDIO_PATH).exists():
        pytest.skip(f"audio file not found: {DEFAULT_AUDIO_PATH}")

    completed = run_experiment_subprocess(
        audio_path=DEFAULT_AUDIO_PATH,
        rounds=1,
        chunk_size=640,
        timeout_sec=180,
    )
    assert completed.returncode == 0, (
        "shared single-instance VAD experiment failed: "
        f"{_format_return_code(completed.returncode)}\n"
        f"stderr:\n{completed.stderr}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
