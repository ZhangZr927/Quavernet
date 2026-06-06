# --- START OF FILE init_utils.py ---
import yaml
import numpy as np

try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:
    from yaml import SafeLoader

from init_config import *

def get_val(d, keys, default=None):
    if not isinstance(d, dict): return default
    for k in keys:
        if k in d: return d[k]
    return default

def get_float_val(d, keys, default=0.0):
    val = get_val(d, keys, default)
    if val is None: return default
    try:
        f_val = float(val)
        if np.isnan(f_val) or np.isinf(f_val): return default
        return f_val
    except (ValueError, TypeError):
        return default

def create_time_mapper(timing_points):
    def ms_to_token(ms):
        if not timing_points: return 0
        prev_ms = get_float_val(timing_points[0], ['StartTime', 'startTime'], 0.0)
        current_bpm = get_float_val(timing_points[0], ['Bpm', 'BPM', 'bpm'], 120.0)
        if current_bpm <= 0: current_bpm = 120.0
        beat = 0.0
        for i in range(len(timing_points)):
            tp = timing_points[i]
            tp_start = get_float_val(tp, ['StartTime', 'startTime'], 0.0)
            tp_bpm = get_float_val(tp, ['Bpm', 'BPM', 'bpm'], 120.0)
            if tp_bpm <= 0: tp_bpm = 120.0
            if ms < tp_start: break
            if i > 0:
                beat += (tp_start - prev_ms) / (60000.0 / current_bpm)
            prev_ms, current_bpm = tp_start, tp_bpm
        if ms > prev_ms:
            beat += (ms - prev_ms) / (60000.0 / current_bpm)
        return int(round(beat * SUBDIVISIONS))
    return ms_to_token

def parse_qua_to_1d_labels(qua_path, num_tokens):
    """把原先的 4 通道降维为 1 通道节奏图"""
    with open(qua_path, 'r', encoding='utf-8') as f:
        parsed = yaml.load(f, Loader=SafeLoader)
        
    timing_points = parsed.get('TimingPoints') or parsed.get('timingPoints') or []
    timing_points = sorted(timing_points, key=lambda x: get_float_val(x, ['StartTime', 'startTime'], 0.0))
    ms_to_token = create_time_mapper(timing_points)
    
    # 变为 (L, 1)
    labels = np.zeros((num_tokens, 1), dtype=np.float32)
    hit_objects = parsed.get('HitObjects') or parsed.get('hitObjects') or []
    
    for obj in hit_objects:
        if not isinstance(obj, dict): continue
        start_ms = get_float_val(obj, ['StartTime', 'startTime'], -1.0)
        if start_ms < 0: continue  
            
        start_token = ms_to_token(start_ms)
        if start_token < num_tokens:
            labels[start_token, 0] = 1.0
            
        end_ms = get_float_val(obj, ['EndTime', 'endTime'], -1.0)
        if end_ms > start_ms:
            end_token = ms_to_token(end_ms)
            if start_token < end_token < num_tokens:
                labels[end_token, 0] = 1.0

    # Target Smearing
    taps = np.where(labels[:, 0] == 1.0)[0]
    for t in taps:
        if t > 0: labels[t-1, 0] = max(labels[t-1, 0], 0.7)
        if t < num_tokens - 1: labels[t+1, 0] = max(labels[t+1, 0], 0.7)
        if t > 1: labels[t-2, 0] = max(labels[t-2, 0], 0.2)
        if t < num_tokens - 2: labels[t+2, 0] = max(labels[t+2, 0], 0.2)
            
    return labels