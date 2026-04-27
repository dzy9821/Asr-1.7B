import os
import time
import torch
import torchaudio
import psutil
import numpy as np

# Add local path to sys.path
import sys
sys.path.append("/home/ubuntu/project/vad-test")

from ten_vad_wrapper import TenVADWrapper

wav_path = "/home/ubuntu/project/vad-test/120报警电话16k.wav"

def measure_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 # MB

print("--- Initial Memory ---")
mem_start = measure_memory()
print(f"Initial Memory: {mem_start:.2f} MB")

import soundfile as sf

# Load Audio
audio_data, sr = sf.read(wav_path, dtype='float32')
if audio_data.ndim > 1:
    audio_data = audio_data[:, 0] # Take first channel
# we assume it is 16kHz based on file name, otherwise resample using scipy
if sr != 16000:
    print(f"Warning: sample rate is {sr}, expected 16000. Skipping resample for simplicity.")

print(f"Audio Length: {len(audio_data)/16000:.2f} seconds")

# 1. Load 20 TenVAD instances
print("\n--- TenVAD 20 Instances Test ---")
mem_before = measure_memory()
ten_vads = []
try:
    for i in range(20):
        ten_vads.append(TenVADWrapper(sample_rate=16000, hop_size=640))
    mem_after = measure_memory()
    print(f"Memory after loading 20 TenVAD instances: {mem_after - mem_before:.2f} MB")
except Exception as e:
    print(f"Failed to load TenVAD: {e}")

if ten_vads:
    start_time = time.time()
    for i in range(20):
        segments = ten_vads[i].get_speech_segments(audio_data, 16000)
    end_time = time.time()
    print(f"Time to process audio with 20 TenVAD instances: {end_time - start_time:.2f} seconds")
    print(f"TenVAD Segments (Instance 0): {segments[:5]} ... (total {len(segments)})")


# 2. Silero VAD Custom Frame Check
print("\n--- Silero VAD Frame Size Check ---")
try:
    model, utils = torch.hub.load(repo_or_dir='/home/ubuntu/project/vad-test/silero-vad', model='silero_vad', source='local', force_reload=True)
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    
    # 40ms = 640 samples
    chunk_640 = torch.zeros(640)
    try:
        res = model(chunk_640, 16000)
        print("Silero VAD successfully processed 640 samples (40ms).")
    except Exception as e:
        print(f"Silero VAD failed to process 640 samples (40ms). Error: {e}")
        
    # Check what happens with standard 512
    try:
        res = model(torch.zeros(512), 16000)
        print("Silero VAD successfully processed 512 samples (32ms).")
    except Exception as e:
        pass
        
except Exception as e:
    print(f"Failed to load Silero VAD: {e}")

# 3. Load 20 Silero VAD instances
print("\n--- Silero VAD 20 Instances Test ---")
mem_before = measure_memory()
silero_vads = []
try:
    for i in range(20):
        m, _ = torch.hub.load(repo_or_dir='/home/ubuntu/project/vad-test/silero-vad', model='silero_vad', source='local')
        silero_vads.append(m)
    mem_after = measure_memory()
    print(f"Memory after loading 20 Silero VAD instances: {mem_after - mem_before:.2f} MB")
    
    audio_tensor = torch.from_numpy(audio_data)
    
    start_time = time.time()
    for i in range(20):
        # process using get_speech_timestamps
        segments = get_speech_timestamps(audio_tensor, silero_vads[i], sampling_rate=16000)
    end_time = time.time()
    print(f"Time to process audio with 20 Silero VAD instances: {end_time - start_time:.2f} seconds")
    print(f"Silero VAD Segments (Instance 0): {segments[:5]} ... (total {len(segments)})")
except Exception as e:
    print(f"Error in Silero 20 instance test: {e}")


# 4. Silero VAD Max Batch / Speed
print("\n--- Silero VAD Max Streams (Batching) Check ---")
# Let's test a single Silero instance with multiple batch sizes to see how many it can handle
try:
    single_silero = silero_vads[0]
    
    batch_sizes = [10, 50, 100, 200, 500, 1000]
    for bs in batch_sizes:
        try:
            # Fake batch of 512 samples (32ms) because Silero might only accept 512 for batching?
            # Let's test with 512
            batch_data = torch.zeros(bs, 512)
            st = time.time()
            single_silero(batch_data, 16000)
            et = time.time()
            print(f"Batch size {bs} processed in {et-st:.4f} seconds (per frame).")
        except Exception as e:
            print(f"Failed at batch size {bs}: {e}")
            break
except Exception as e:
    print(f"Error in Batching test: {e}")

