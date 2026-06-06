# --- START OF FILE stage2_infer.py ---
import os
import yaml
import torch
import librosa
import numpy as np
import scipy.signal
from stage2_config import DEVICE, CHUNK_SIZE, SUBDIVISIONS
from stage2_model import RhythmNet, MaskedChartGenerator

DEVICE_INFER = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONSTANT_OFFSET_SEC = -0.035  

# =========================================================================
# 搬运自原 autochart3 的音频与特征提取逻辑，使其不再依赖旧文件
# =========================================================================
def auto_detect_timing_points(audio_path):
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
    timing_points = []
    if len(beat_times) < 6:
        bpm = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
        first_beat = max(0.0, (beat_times[0] + CONSTANT_OFFSET_SEC)) if len(beat_times) > 0 else 0.0
        return y, sr, [{'StartTime': int(first_beat * 1000), 'Bpm': float(round(bpm, 2))}]

    beats = np.maximum(0.0, beat_times + CONSTANT_OFFSET_SEC).tolist()
    beats = [b for i, b in enumerate(beats) if i == 0 or b > beats[i-1]]
    
    curr_idx = 0
    TOLERANCE = 0.025  

    while curr_idx < len(beats):
        if curr_idx >= len(beats) - 1:
            bpm = timing_points[-1]['Bpm'] if timing_points else 120.0
            timing_points.append({'StartTime': beats[-1] * 1000, 'Bpm': bpm})
            break
            
        lookahead = beats[curr_idx : min(curr_idx + 8, len(beats))]
        diffs = np.diff(lookahead)
        median_tick = float(np.median(diffs)) if len(diffs) >= 3 else float(np.mean(diffs))
            
        start_time = beats[curr_idx]
        end_offset = curr_idx
        best_tick = median_tick
        
        for i in range(curr_idx + 1, len(beats)):
            elapsed = beats[i] - start_time
            logical_beats = round(elapsed / median_tick)
            if logical_beats <= 0: break
                
            test_tick = elapsed / logical_beats
            max_dev = 0.0
            for j in range(curr_idx + 1, i + 1):
                inner_elapsed = beats[j] - start_time
                inner_logical = round(inner_elapsed / median_tick)
                projected_time = inner_logical * test_tick
                max_dev = max(max_dev, abs(inner_elapsed - projected_time))
                
            if max_dev > TOLERANCE or abs(test_tick - median_tick) / median_tick > 0.05:
                break
            end_offset = i
            best_tick = test_tick
        
        bpm = 60.0 / best_tick if best_tick > 0.1 else 120.0
        while start_time < 0: start_time += best_tick
            
        timing_points.append({'StartTime': start_time * 1000, 'Bpm': bpm})
        curr_idx = end_offset + 1

    final_tps = []
    for tp in timing_points:
        t_ms = int(round(tp['StartTime']))
        t_bpm = float(round(tp['Bpm'], 2))
        if final_tps and abs(t_ms - final_tps[-1]['StartTime']) < 10:
            continue
        final_tps.append({'StartTime': t_ms, 'Bpm': t_bpm})
        
    return y, sr, final_tps

def generate_timestamps_from_timing_points(timing_points, audio_duration_sec, subdivisions):
    timestamps = []
    if not timing_points: return np.array([])
    
    first_tp = timing_points[0]
    t_start = first_tp['StartTime'] / 1000.0
    tick = (60.0 / first_tp['Bpm']) / subdivisions
    
    t = t_start - tick
    pre = []
    while t >= 0:
        pre.append(t)
        t -= tick
    timestamps.extend(reversed(pre))
    
    for i in range(len(timing_points)):
        start_sec = timing_points[i]['StartTime'] / 1000.0
        bpm = timing_points[i]['Bpm']
        tick_sec = (60.0 / bpm) / subdivisions
        end_sec = timing_points[i+1]['StartTime'] / 1000.0 if i + 1 < len(timing_points) else audio_duration_sec
        
        idx = 0
        while True:
            current_t = start_sec + idx * tick_sec
            if current_t >= end_sec - 1e-5: break
            timestamps.append(current_t)
            idx += 1
            
    return np.array(timestamps)

def extract_features(y, sr, timing_points):
    audio_duration_sec = librosa.get_duration(y=y, sr=sr)
    timestamps = generate_timestamps_from_timing_points(timing_points, audio_duration_sec, SUBDIVISIONS)

    mel_basis = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=80, fmin=20, fmax=sr/2)
    window = scipy.signal.get_window('hann', 2048)
    sample_indices = np.round(timestamps * sr).astype(int)
    pad_length = 2048 // 2
    y_padded = np.pad(y, pad_length, mode='reflect')

    spectrogram, rms_list = [], []
    for idx in sample_indices:
        if idx < 0 or idx >= len(y):
            spectrogram.append(np.zeros(80))
            rms_list.append(0.0)
            continue
        frame = y_padded[idx : idx+2048]
        stft_frame = np.fft.rfft(frame * window)
        spectrogram.append(np.dot(mel_basis, np.abs(stft_frame)**2))
        rms_list.append(np.sqrt(np.mean(frame**2)))

    log_mel_spec = librosa.power_to_db(np.array(spectrogram), ref=np.max)
    log_mel_spec = np.clip((log_mel_spec + 80.0) / 80.0, 0.0, 1.0)
    rms_db = librosa.amplitude_to_db(np.array(rms_list), ref=np.max)
    return log_mel_spec, timestamps, rms_db

def generate_stage2_qua(probs_4d, peak_indices, timestamps, timing_points, target_diff, mp3_filename, output_qua_path, rms_db):
    new_hit_objects = []
    SILENCE_THRESHOLD = -55.0  

    if target_diff < 1.5: max_chord = 1        
    elif target_diff <= 3.5: max_chord = 2     
    elif target_diff <= 6.5: max_chord = 3
    else: max_chord = 4

    rejected_count = 0

    for peak_token in peak_indices:
        if peak_token >= len(rms_db) or rms_db[peak_token] < SILENCE_THRESHOLD:
            continue
            
        lane_probs = probs_4d[peak_token] 
        
        # 拒绝判决
        if np.max(lane_probs) < 0.2: 
            rejected_count += 1
            continue

        valid_lanes = []
        for lane in range(4):
            if lane_probs[lane] >= 0.3:
                valid_lanes.append((lane, lane_probs[lane]))
                
        valid_lanes = sorted(valid_lanes, key=lambda x: x[1], reverse=True)[:max_chord]
        start_ms = timestamps[peak_token] * 1000.0
        
        for actual_lane, _ in valid_lanes:
            new_hit_objects.append({
                'StartTime': int(round(start_ms)),
                'Lane': int(actual_lane) + 1
            })

    new_hit_objects.sort(key=lambda x: x['StartTime'])

    qua_dict = {
        'AudioFile': mp3_filename,
        'Song': 'AI Generated (2-Stage)',
        'Artist': 'QuaverNet AI',
        'Creator': 'AI Charter',
        'DifficultyName': f'AI Diff {target_diff:.1f}',
        'MapId': -1,
        'MapSetId': -1,
        'Mode': 'Keys4',
        'TimingPoints': timing_points,
        'HitObjects': new_hit_objects
    }

    with open(output_qua_path, 'w', encoding='utf-8') as f:
        yaml.dump(qua_dict, f, sort_keys=False)

    print(f"[*] 🌟 2-Stage 谱面生成成功! 难度: {target_diff:.1f}")
    print(f"[*] 一阶段提供节奏点: {len(peak_indices)} 个, 二阶段主动拒绝(修正): {rejected_count} 个。")
    print(f"[*] 最终落盘按键总数: {len(new_hit_objects)} 个。")
    print(f"[*] 保存在: {output_qua_path}")


def main_infer(mp3_path, output_dir, target_diff=2.5, 
               stage1_path="init_models/best_rhythm_model.pth", 
               stage2_path="stage2_models/best_stage2_generator.pth"):
    
    os.makedirs(output_dir, exist_ok=True)
    mp3_filename = os.path.basename(mp3_path)
    output_qua_path = os.path.join(output_dir, mp3_filename.replace('.mp3', f'_S2_diff_{target_diff:.1f}.qua'))

    print("[*] 1. 读取音频并解析 Timing...")
    y, sr, timing_points = auto_detect_timing_points(mp3_path)
    feature, timestamps, rms_db = extract_features(y, sr, timing_points)
    L = feature.shape[0]

    print("[*] 2. 加载一阶段模型提取节奏规律...")
    rhythm_model = RhythmNet().to(DEVICE_INFER)
    rhythm_model.load_state_dict(torch.load(stage1_path, map_location=DEVICE_INFER))
    rhythm_model.eval()
    
    probs_1d_list = []
    diff_tensor = torch.tensor([target_diff], dtype=torch.float32).unsqueeze(0).to(DEVICE_INFER)
    
    for start_idx in range(0, L, CHUNK_SIZE):
        end_idx = min(start_idx + CHUNK_SIZE, L)
        feat_chunk = feature[start_idx:end_idx]
        pad_len = CHUNK_SIZE - feat_chunk.shape[0]
        if pad_len > 0: feat_chunk = np.pad(feat_chunk, ((0, pad_len), (0, 0)), mode='constant')

        X = torch.tensor(feat_chunk, dtype=torch.float32).transpose(0, 1).unsqueeze(0).to(DEVICE_INFER)
        with torch.no_grad():
            logits_1d = rhythm_model(X, diff_tensor)
            probs_chunk = torch.sigmoid(logits_1d).squeeze(0).cpu().numpy()
        
        if pad_len > 0: probs_chunk = probs_chunk[:-pad_len]
        probs_1d_list.append(probs_chunk)

    probs_1d_full = np.concatenate(probs_1d_list, axis=0).flatten()

    sharp_probs = probs_1d_full ** 3
    dyn_distance = max(2, int(6.0 - (target_diff - 1.0) * 0.4)) 
    dyn_prominence = max(0.01, 0.10 - (target_diff - 1.0) * 0.012)
    dyn_height = max(0.05, 0.20 - (target_diff - 1.0) * 0.015) 
    
    peak_indices, _ = scipy.signal.find_peaks(
        sharp_probs, height=dyn_height, distance=dyn_distance, prominence=dyn_prominence
    )
    
    mask_full = np.zeros((L, 1), dtype=np.float32)
    mask_full[peak_indices, 0] = 1.0

    print("[*] 3. 加载二阶段模型进行掩码受限排键...")
    pattern_model = MaskedChartGenerator().to(DEVICE_INFER)
    pattern_model.load_state_dict(torch.load(stage2_path, map_location=DEVICE_INFER))
    pattern_model.eval()
    
    probs_4d_list = []
    for start_idx in range(0, L, CHUNK_SIZE):
        end_idx = min(start_idx + CHUNK_SIZE, L)
        feat_chunk = feature[start_idx:end_idx]
        mask_chunk = mask_full[start_idx:end_idx]
        
        pad_len = CHUNK_SIZE - feat_chunk.shape[0]
        if pad_len > 0: 
            feat_chunk = np.pad(feat_chunk, ((0, pad_len), (0, 0)), mode='constant')
            mask_chunk = np.pad(mask_chunk, ((0, pad_len), (0, 0)), mode='constant')

        X = torch.tensor(feat_chunk, dtype=torch.float32).transpose(0, 1).unsqueeze(0).to(DEVICE_INFER)
        M = torch.tensor(mask_chunk, dtype=torch.float32).transpose(0, 1).unsqueeze(0).to(DEVICE_INFER) 
        
        with torch.no_grad():
            logits_4d = pattern_model(X, diff_tensor, M)
            probs_4d_chunk = torch.sigmoid(logits_4d).squeeze(0).cpu().numpy()
            
        if pad_len > 0: probs_4d_chunk = probs_4d_chunk[:-pad_len]
        probs_4d_list.append(probs_4d_chunk)

    probs_4d_full = np.concatenate(probs_4d_list, axis=0)
    
    print("[*] 4. 执行后处理装载...")
    generate_stage2_qua(probs_4d_full, peak_indices, timestamps, timing_points, target_diff, mp3_filename, output_qua_path, rms_db)


if __name__ == "__main__":
    TEST_MP3 = "see you again.mp3"
    OUTPUT_FOLDER = "ai_2stage_chart"
    TEST_TARGET_DIFF = 3.5
    
    if not os.path.exists(TEST_MP3):
        print(f"❌ 找不到测试音频: {TEST_MP3}")
    else:
        main_infer(TEST_MP3, OUTPUT_FOLDER, target_diff=TEST_TARGET_DIFF)