# --- START OF FILE init_train.py ---
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from init_config import *
from init_dataset import get_dataloaders
from init_model import RhythmNet

class WeightedFocalLoss(nn.Module):
    """为了应对 1/48 分辨率下大量没有音符的背景帧而引入 Focal Loss"""
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss) 
        weight = torch.ones_like(targets)
        weight[targets > 0.5] = self.pos_weight
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss * weight
        return focal_loss.mean()

def calculate_tolerant_metrics(preds, trues, tolerance=4):
    """沿用容忍度矩阵: +/- 4 帧(即正负大几十毫秒)内的击中都算正确"""
    B, L, C = preds.shape
    preds_pool = preds.permute(0, 2, 1).reshape(B * C, 1, L)
    trues_pool = trues.permute(0, 2, 1).reshape(B * C, 1, L)
    
    kernel_size = 2 * tolerance + 1
    trues_dilated = F.max_pool1d(trues_pool, kernel_size=kernel_size, stride=1, padding=tolerance)
    preds_dilated = F.max_pool1d(preds_pool, kernel_size=kernel_size, stride=1, padding=tolerance)
    
    tp_precision = (preds_pool.view(-1) * trues_dilated.view(-1)).sum().item()
    tp_recall = (trues_pool.view(-1) * preds_dilated.view(-1)).sum().item()
    
    return tp_precision, tp_recall, preds_pool.sum().item(), trues_pool.sum().item()

def main():
    print(f"[*] Pre-training with device: {DEVICE}")
    train_loader, val_loader = get_dataloaders()
    model = RhythmNet().to(DEVICE)
    
    criterion = WeightedFocalLoss(pos_weight=POS_WEIGHT)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    
    best_f1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")
        for X, Y, diff in pbar:
            X, Y, diff = X.to(DEVICE), Y.to(DEVICE), diff.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X, diff)
            
            loss = criterion(logits, Y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        total_tp_prec, total_tp_rec, total_pred, total_true = 0, 0, 0, 0
        
        with torch.no_grad():
            for X, Y, diff in val_loader:
                X, Y, diff = X.to(DEVICE), Y.to(DEVICE), diff.to(DEVICE)
                logits = model(X, diff)
                val_loss += criterion(logits, Y).item()
                
                # 保守阈值 -0.4
                preds = (logits > -0.4).float()
                trues = (Y > 0.5).float() 
                
                tp_p, tp_r, t_p, t_t = calculate_tolerant_metrics(preds, trues, tolerance=4)
                total_tp_prec += tp_p
                total_tp_rec += tp_r
                total_pred += t_p
                total_true += t_t
                
        val_loss /= len(val_loader)
        
        t_precision = total_tp_prec / (total_pred + 1e-8)
        t_recall = total_tp_rec / (total_true + 1e-8)
        t_f1 = 2 * (t_precision * t_recall) / (t_precision + t_recall + 1e-8)
        
        print(f"--> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"--> [Tolerant] Recall: {t_recall:.4f} | Precision: {t_precision:.4f} | F1: {t_f1:.4f}")
        
        if t_f1 > best_f1:
            best_f1 = t_f1
            save_path = os.path.join(MODELS_DIR, "best_rhythm_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"[*] 🌟 Best model saved! (Tolerant F1: {best_f1:.4f})")

if __name__ == "__main__":
    main()