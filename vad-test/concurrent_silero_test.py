import time
import torch
import soundfile as sf

wav_path = "/home/ubuntu/project/vad-test/120报警电话16k.wav"

def custom_batched_get_speech_timestamps(
    audio: torch.Tensor,
    model,
    batch_size: int,
    threshold: float = 0.5,
    sampling_rate: int = 16000,
    min_speech_duration_ms: int = 250,
    max_speech_duration_s: float = float('inf'),
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
    min_silence_at_max_speech: int = 98,
    use_max_poss_sil_at_max_speech: bool = True
):
    window_size_samples = 512
    model.reset_states() # Reset state (no arguments for JIT model)

    min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
    speech_pad_samples = sampling_rate * speech_pad_ms / 1000
    max_speech_samples = sampling_rate * max_speech_duration_s - window_size_samples - 2 * speech_pad_samples
    min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
    min_silence_samples_at_max_speech = sampling_rate * min_silence_at_max_speech / 1000

    audio_length_samples = len(audio)
    speech_probs = []
    
    start_time = time.time()
    
    # Process audio chunk by chunk, simulating batched stream
    for current_start_sample in range(0, audio_length_samples, window_size_samples):
        chunk = audio[current_start_sample: current_start_sample + window_size_samples]
        if len(chunk) < window_size_samples:
            chunk = torch.nn.functional.pad(chunk, (0, int(window_size_samples - len(chunk))))
        
        # Simulate batching N streams
        chunk_batch = chunk.unsqueeze(0).repeat(batch_size, 1)
        
        # Forward pass
        prob_batch = model(chunk_batch, sampling_rate)
        
        # Extract prob for the first stream (since all streams are identical in this test)
        speech_prob = prob_batch[0].item()
        speech_probs.append(speech_prob)

    process_time = time.time() - start_time

    # Construct segments based on Silero's logic
    triggered = False
    speeches = []
    current_speech = {}
    neg_threshold = max(threshold - 0.15, 0.01)
    temp_end = 0
    prev_end = next_start = 0
    possible_ends = []

    for i, speech_prob in enumerate(speech_probs):
        cur_sample = window_size_samples * i

        if (speech_prob >= threshold) and temp_end:
            sil_dur = cur_sample - temp_end
            if sil_dur > min_silence_samples_at_max_speech:
                possible_ends.append((temp_end, sil_dur))
            temp_end = 0
            if next_start < prev_end:
                next_start = cur_sample

        if (speech_prob >= threshold) and not triggered:
            triggered = True
            current_speech['start'] = cur_sample
            continue

        if triggered and (cur_sample - current_speech['start'] > max_speech_samples):
            if use_max_poss_sil_at_max_speech and possible_ends:
                prev_end, dur = max(possible_ends, key=lambda x: x[1])
                current_speech['end'] = prev_end
                speeches.append(current_speech)
                current_speech = {}
                next_start = prev_end + dur

                if next_start < prev_end + cur_sample:
                    current_speech['start'] = next_start
                else:
                    triggered = False
                prev_end = next_start = temp_end = 0
                possible_ends = []
            else:
                if prev_end:
                    current_speech['end'] = prev_end
                    speeches.append(current_speech)
                    current_speech = {}
                    if next_start < prev_end:
                        triggered = False
                    else:
                        current_speech['start'] = next_start
                    prev_end = next_start = temp_end = 0
                    possible_ends = []
                else:
                    current_speech['end'] = cur_sample
                    speeches.append(current_speech)
                    current_speech = {}
                    prev_end = next_start = temp_end = 0
                    triggered = False
                    possible_ends = []
                    continue

        if (speech_prob < neg_threshold) and triggered:
            if not temp_end:
                temp_end = cur_sample
            sil_dur_now = cur_sample - temp_end

            if not use_max_poss_sil_at_max_speech and sil_dur_now > min_silence_samples_at_max_speech:
                prev_end = temp_end

            if sil_dur_now < min_silence_samples:
                continue
            else:
                current_speech['end'] = temp_end
                if (current_speech['end'] - current_speech['start']) > min_speech_samples:
                    speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                possible_ends = []
                continue

    if current_speech and (audio_length_samples - current_speech['start']) > min_speech_samples:
        current_speech['end'] = audio_length_samples
        speeches.append(current_speech)

    for i, speech in enumerate(speeches):
        if i == 0:
            speech['start'] = int(max(0, speech['start'] - speech_pad_samples))
        if i != len(speeches) - 1:
            silence_duration = speeches[i+1]['start'] - speech['end']
            if silence_duration < 2 * speech_pad_samples:
                speech['end'] += int(silence_duration // 2)
                speeches[i+1]['start'] = int(max(0, speeches[i+1]['start'] - silence_duration // 2))
            else:
                speech['end'] = int(min(audio_length_samples, speech['end'] + speech_pad_samples))
                speeches[i+1]['start'] = int(max(0, speeches[i+1]['start'] - speech_pad_samples))
        else:
            speech['end'] = int(min(audio_length_samples, speech['end'] + speech_pad_samples))

    return speeches, process_time

def main():
    print("Loading audio...")
    audio_data, sr = sf.read(wav_path, dtype='float32')
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]
    audio_tensor = torch.from_numpy(audio_data)

    print("Loading Silero VAD model...")
    model, utils = torch.hub.load(repo_or_dir='/home/ubuntu/project/vad-test/silero-vad', model='silero_vad', source='local')
    
    concurrency_levels = [1, 20, 50, 100]
    
    with open("silero_concurrent_results.txt", "w", encoding="utf-8") as f:
        f.write(f"Silero VAD 测试结果\n")
        f.write(f"测试音频: {wav_path}\n")
        f.write(f"音频时长: {len(audio_tensor)/16000:.2f} 秒\n\n")
        
        for n in concurrency_levels:
            print(f"Testing {n} concurrent streams...")
            segments, proc_time = custom_batched_get_speech_timestamps(
                audio=audio_tensor,
                model=model,
                batch_size=n,
                sampling_rate=16000
            )
            
            # Write to file
            f.write(f"=== 并发流数量 (Batch Size): {n} ===\n")
            f.write(f"处理耗时: {proc_time:.4f} 秒\n")
            rtf = proc_time / (len(audio_tensor)/16000)
            f.write(f"RTF (实时率，每路): {rtf/n:.4f}\n")
            f.write(f"分段数量: {len(segments)}\n")
            f.write("分段时间戳 (采样点):\n")
            for seg in segments:
                f.write(f"  - start: {seg['start']}, end: {seg['end']}\n")
            f.write("\n")

if __name__ == "__main__":
    main()
