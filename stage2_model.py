# --- START OF FILE stage2_model.py ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from stage2_config import *
from init_model import FiLMLayer, ResBlock1D, RhythmNet 

class DiffEmbedding(nn.Module):
    def __init__(self, out_channels=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 8),
            nn.LeakyReLU(0.2),
            nn.Linear(8, out_channels)
        )
    def forward(self, diff, seq_len):
        emb = self.net(diff)
        return emb.unsqueeze(2).expand(-1, -1, seq_len)

class ConvBlock1D(nn.Module):
    def __init__(self, in_c, out_c, dilation=1):
        super().__init__()
        padding = 2 * dilation
        self.conv1 = weight_norm(nn.Conv1d(in_c, out_c, kernel_size=5, padding=padding, dilation=dilation))
        self.gn1 = nn.GroupNorm(8, out_c)
        self.conv2 = weight_norm(nn.Conv1d(out_c, out_c, kernel_size=5, padding=2))
        self.gn2 = nn.GroupNorm(8, out_c)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        res = x
        x = self.relu(self.gn1(self.conv1(x)))
        x = self.relu(self.gn2(self.conv2(x)))
        if x.shape[1] == res.shape[1]: x = x + res
        return x

# ================= 2阶段: 掩码生成器 =================
class MaskedChartGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.diff_emb = DiffEmbedding(16)
        
        # 输入为 Audio(80) + Diff(16) + Mask(1) = 97
        self.enc1 = ConvBlock1D(97, 64)                
        self.enc2 = ConvBlock1D(64, 128)               
        self.enc3 = ConvBlock1D(128, 256)              
        self.enc4 = ConvBlock1D(256, 512)              
        
        self.pool = nn.MaxPool1d(2)
        
        self.bottleneck = nn.Sequential(
            ConvBlock1D(512, 512, dilation=2),
            ConvBlock1D(512, 512, dilation=4),
            ConvBlock1D(512, 512, dilation=8),
            ConvBlock1D(512, 512, dilation=16) 
        )
        
        self.dec3 = ConvBlock1D(512 + 256, 256)
        self.dec2 = ConvBlock1D(256 + 128, 128)
        self.dec1 = ConvBlock1D(128 + 64, 64)
        
        self.final_conv = weight_norm(nn.Conv1d(64, OUT_CLASSES, kernel_size=3, padding=1))

    def forward(self, x, diff, mask):
        B, _, L = x.size()
        d_emb = self.diff_emb(diff, L)
        
        # 将 Mask 拼入特征，让网络明确知道哪里需要排布音符
        c = torch.cat([x, d_emb, mask], dim=1) 
        
        e1 = self.enc1(c)           
        e2 = self.enc2(self.pool(e1)) 
        e3 = self.enc3(self.pool(e2)) 
        e4 = self.enc4(self.pool(e3)) 
        b = self.bottleneck(e4)       
        
        d3 = F.interpolate(b, scale_factor=2, mode='nearest')
        d3 = self.dec3(torch.cat([d3, e3], dim=1)) 
        d2 = F.interpolate(d3, scale_factor=2, mode='nearest')
        d2 = self.dec2(torch.cat([d2, e2], dim=1)) 
        d1 = F.interpolate(d2, scale_factor=2, mode='nearest')
        d1 = self.dec1(torch.cat([d1, e1], dim=1)) 
        
        logits = self.final_conv(d1).permute(0, 2, 1) # (B, L, 4)
        
        # ==========================================================
        # 【核心物理限制】
        # 将 mask 为 0 的位置 logits 强行拉向负无穷
        # 这里 mask 需要从 (B, 1, L) 转换为 (B, L, 1) 对齐 logits
        # ==========================================================
        mask_aligned = mask.permute(0, 2, 1) 
        logits = logits - (1.0 - mask_aligned) * 10000.0
        
        return logits

# ================= 2阶段: 鉴别器 (与原版保持完全一致) =================
class ConditionalDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.diff_emb = DiffEmbedding(16)
        # 输入： 80(Audio) + 4(Chart) + 16(Diff) = 100
        self.net = nn.Sequential(
            weight_norm(nn.Conv1d(100, 128, kernel_size=7, stride=2, padding=3)),
            nn.LeakyReLU(0.2),
            weight_norm(nn.Conv1d(128, 256, kernel_size=7, stride=2, padding=3)),
            nn.LeakyReLU(0.2),
            weight_norm(nn.Conv1d(256, 512, kernel_size=7, stride=2, padding=3)),
            nn.LeakyReLU(0.2),
            weight_norm(nn.Conv1d(512, 1024, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            weight_norm(nn.Linear(1024, 1))
        )

    def forward(self, audio, chart, diff):
        L = chart.size(1)
        chart = chart.permute(0, 2, 1)
        d_emb = self.diff_emb(diff, L)
        c = torch.cat([audio, chart, d_emb], dim=1) 
        return self.net(c)