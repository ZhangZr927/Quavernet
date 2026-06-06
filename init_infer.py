# --- START OF FILE init_infer.py ---
# 用于测试init_model，将把所有按键放在第2通道以测试节奏。
import os
import yaml
import torch
import librosa
import numpy as np
import scipy.signal
from init_model import RhythmNet
from init_config import DEVICE, CHUNK_SIZE, SUBDIVISIONS

from autochart3 import auto_detect_timing_points, extract_features

# ================================
DEVICE_INFER = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ================================

def generate_1lane_qua(probs_1d, timestamps, timing_points, target_diff, mp3_filename, output_qua_path, rms_db):
    new_hit_objects = []
    SILENCE_THRESHOLD = -55.0  

    # 【预留：信号锐化】（未采用）
    sharp_probs = probs_1d #** 3

    # 【预留：动态地形参数】
    dyn_distance = 3
    dyn_prominence = 0
    dyn_height = 0.5

    print(f"  [Auto-Config] Distance: {dyn_distance}, Prominence: {dyn_prominence:.3f}, Height: {dyn_height:.3f}")

    peaks, _ = scipy.signal.find_peaks(
        sharp_probs, 
        height=dyn_height, 
        distance=dyn_distance, 
        prominence=dyn_prominence
    )
    
    for peak_token in peaks:
        if peak_token < len(rms_db) and rms_db[peak_token] < SILENCE_THRESHOLD:
            continue
        
        start_ms = timestamps[peak_token] * 1000.0
        new_hit_objects.append({
            'StartTime': int(round(start_ms)),
            'Lane': 2
        })

    new_hit_objects.sort(key=lambda x: x['StartTime'])

    qua_dict = {
        'AudioFile': mp3_filename,
        'Song': 'AI PreTrain Rhythm Test',
        'Artist': 'QuaverNet AI',
        'Creator': 'AI Charter',
        'DifficultyName': f'Rhythm Test Diff {target_diff:.1f}',
        'MapId': -1,
        'MapSetId': -1,
        'Mode': 'Keys4',
        'TimingPoints': timing_points,
        'HitObjects': new_hit_objects
    }

    with open(output_qua_path, 'w', encoding='utf-8') as f:
        yaml.dump(qua_dict, f, sort_keys=False)

    print(f"[*] 🥁 纯节奏测试谱面生成成功! 难度: {target_diff:.1f}, 产生节奏点 {len(new_hit_objects)} 个。")
    print(f"[*] 保存在: {output_qua_path}")


def main_infer(mp3_path, output_dir, target_diff=2.5, model_path="models/best_rhythm_model.pth"):
    os.makedirs(output_dir, exist_ok=True)
    mp3_filename = os.path.basename(mp3_path)
    output_qua_path = os.path.join(output_dir, mp3_filename.replace('.mp3', f'_RHYTHM_{target_diff:.1f}.qua'))

    print("[*] 读取音频并划分区块...")
    y, sr, timing_points = auto_detect_timing_points(mp3_path)
    feature, timestamps, rms_db = extract_features(y, sr, timing_points)

    print("[*] 正在加载基础节奏判断模型...")
    model = RhythmNet().to(DEVICE_INFER)
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE_INFER))
    except Exception as e:
        print(f"❌ 加载节奏模型失败: {e}")
        return

    model.eval()
    L = feature.shape[0]
    probs_list = []

    for start_idx in range(0, L, CHUNK_SIZE):
        end_idx = min(start_idx + CHUNK_SIZE, L)
        feat_chunk = feature[start_idx:end_idx]

        pad_len = CHUNK_SIZE - feat_chunk.shape[0]
        if pad_len > 0:
            feat_chunk = np.pad(feat_chunk, ((0, pad_len), (0, 0)), mode='constant')

        X = torch.tensor(feat_chunk, dtype=torch.float32).transpose(0, 1).unsqueeze(0).to(DEVICE_INFER)
        diff_tensor = torch.tensor([target_diff], dtype=torch.float32).unsqueeze(0).to(DEVICE_INFER)

        with torch.no_grad():
            logits = model(X, diff_tensor)
            probs_chunk = torch.sigmoid(logits).squeeze(0).cpu().numpy() # Shape: (CHUNK_SIZE, 1)

        if pad_len > 0:
            probs_chunk = probs_chunk[:-pad_len]
        probs_list.append(probs_chunk)

    final_probs = np.concatenate(probs_list, axis=0).flatten() # (L,)
    
    generate_1lane_qua(final_probs, timestamps, timing_points, target_diff, mp3_filename, output_qua_path, rms_db)


if __name__ == "__main__":
    TEST_MP3 = "cruel_summer.mp3"
    OUTPUT_FOLDER = "ai_rhythm_test"
    TEST_TARGET_DIFF = 3
    
    if not os.path.exists(TEST_MP3):
        print(f"❌ 找不到测试音频: {TEST_MP3}")
    else:
        main_infer(TEST_MP3, OUTPUT_FOLDER, target_diff=TEST_TARGET_DIFF)