# --- START OF FILE dataset.py ---
import os
import re
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from utils import parse_qua_to_labels
from stage2_config import *

# 将处理单个样本的逻辑提取到顶层，以便支持多进程并发序列化
def _load_single_sample(npy_path, chunk_size):
    qua_path = re.sub(r'_diff_[\d\.]+_spec\.npy$', '.qua', npy_path)
    if not os.path.exists(qua_path): return None
    
    diff = 2.0 # 默认保护值
    match = re.search(r'_diff_([\d\.]+)_spec\.npy$', npy_path)
    if match:
        diff = float(match.group(1))
        
    feature = np.load(npy_path).astype(np.float32)
    feature = (feature + 80.0) / 80.0 
    feature = np.clip(feature, 0.0, 1.0)
    
    L = feature.shape[0]
    if L < chunk_size: return None  
    
    # 这一步是耗时大头，会被多核并行处理
    labels = parse_qua_to_labels(qua_path, L)
    
    return {'feature': feature, 'label': labels, 'length': L, 'diff': diff}


class QuaverDataset(Dataset):
    def __init__(self, data_dir, is_train=True, val_split=0.1):
        self.samples = []
        self.is_train = is_train
        self.epoch_multiplier = 5 if is_train else 1
        
        npy_files = glob.glob(os.path.join(data_dir, "**/*_spec.npy"), recursive=True)
        npy_files = sorted(npy_files)
        
        np.random.seed(42) 
        np.random.shuffle(npy_files)
        split_idx = int(len(npy_files) * (1 - val_split))
        files_to_use = npy_files[:split_idx] if is_train else npy_files[split_idx:]
        
        set_name = "Train" if is_train else "Validation"
        print(f"\n[*] 正在加载并解析 {set_name} 数据集 (共 {len(files_to_use)} 首)...")
        
        # 使用 CPU 多核并行加载数据与解析 YAML
        max_workers = max(1, os.cpu_count() - 2)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_load_single_sample, p, CHUNK_SIZE): p for p in files_to_use}
            
            # tqdm 进度条
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"{set_name} Load"):
                res = future.result()
                if res is not None:
                    self.samples.append(res)
                    
        print(f"[*] {set_name} set 初始化完成. 成功载入 {len(self.samples)} 个有效样本。\n")

    def __len__(self):
        return len(self.samples) * self.epoch_multiplier

    def __getitem__(self, idx):
        real_idx = idx % len(self.samples)
        sample = self.samples[real_idx]
        feat = sample['feature']
        label = sample['label']
        L = sample['length']
        diff = sample['diff']
        
        if self.is_train:
            start_idx = np.random.randint(0, L - CHUNK_SIZE + 1)
        else:
            start_idx = (L - CHUNK_SIZE) // 2 
            
        feat_crop = feat[start_idx : start_idx + CHUNK_SIZE] 
        label_crop = label[start_idx : start_idx + CHUNK_SIZE].copy() 
        
        feat_crop = torch.tensor(feat_crop).transpose(0, 1) # (80, L)
        label_crop = torch.tensor(label_crop)               # (L, 4)
        diff_tensor = torch.tensor([diff], dtype=torch.float32)
        
        if self.is_train and torch.rand(1).item() > 0.4: 
            shift = torch.randint(1, 4, (1,)).item()
            perm = (torch.arange(4) + shift) % 4
            label_crop = label_crop[:, perm]

        return feat_crop, label_crop, diff_tensor

def get_dataloaders():
    train_ds = QuaverDataset(DATA_DIR, is_train=True, val_split=VALIDATION_SPLIT)
    val_ds = QuaverDataset(DATA_DIR, is_train=False, val_split=VALIDATION_SPLIT)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader