# --- START OF FILE init_model.py ---
import torch
import torch.nn as nn
from init_config import *

# FiLM 调节器
class FiLMLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 根据一个单独的浮点数(难度)生成每个通道的 gamma 和 beta
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, channels * 2)
        )
        
    def forward(self, x, diff):
        # x: (Batch, channels, L), diff: (Batch, 1)
        stats = self.net(diff).unsqueeze(2)
        gamma, beta = stats.chunk(2, dim=1)
        # 对特征进行尺度缩放和偏移，使难度不被 Norm 抵消！
        return x * (1 + gamma) + beta

class ResBlock1D(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        padding = ((CNN_KERNEL_SIZE - 1) // 2) * dilation
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=CNN_KERNEL_SIZE, padding=padding, dilation=dilation)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=CNN_KERNEL_SIZE, padding=padding, dilation=dilation)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return self.relu(out + residual)

class RhythmNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 输入直接是音频的 80 通道
        self.conv_in = nn.Conv1d(IN_CHANNELS, CNN_CHANNELS, kernel_size=CNN_KERNEL_SIZE, padding=CNN_KERNEL_SIZE//2)
        self.gn_in = nn.GroupNorm(8, CNN_CHANNELS)
        self.relu = nn.ReLU()
        
        # FiLM 层
        self.film_in = FiLMLayer(CNN_CHANNELS)
        
        self.res_blocks = nn.ModuleList([
            ResBlock1D(CNN_CHANNELS, dilation=1),
            ResBlock1D(CNN_CHANNELS, dilation=2),
            ResBlock1D(CNN_CHANNELS, dilation=4)
        ])
        
        self.film_deep = FiLMLayer(CNN_CHANNELS) # 进 LSTM 前再强调一次难度
        
        self.bilstm = nn.LSTM(
            input_size=CNN_CHANNELS,
            hidden_size=RNN_HIDDEN_SIZE,
            num_layers=RNN_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=RNN_HIDDEN_SIZE * 2, 
            num_heads=ATTENTION_HEADS, 
            batch_first=True,
            dropout=DROPOUT
        )
        self.ln = nn.LayerNorm(RNN_HIDDEN_SIZE * 2)
        
        fc_in_dim = (RNN_HIDDEN_SIZE * 2) + CNN_CHANNELS
        
        self.fc = nn.Sequential(
            nn.Linear(fc_in_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(DROPOUT),  
            nn.Linear(128, OUT_CLASSES)
        )

    def forward(self, x, diff):
        # x: (B, 80, L), diff: (B, 1)
        
        # 1. 提取原始高频特征
        c = self.relu(self.gn_in(self.conv_in(x)))
        
        # 2. 打上强烈的难度标记！
        early_features = self.film_in(c, diff)
        
        c = early_features
        for block in self.res_blocks:
            c = block(c)
            
        # 再次强调难度，防止被长残差稀释
        c = self.film_deep(c, diff)
        
        c = c.permute(0, 2, 1)  # (Batch, L, Channels)
        
        # 3. 宏观依赖特征提取
        lstm_out, _ = self.bilstm(c)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        deep_features = self.ln(lstm_out + attn_out) 
        
        # 4. 跳跃拼接
        early_anchors = early_features.permute(0, 2, 1)
        fused_features = torch.cat([deep_features, early_anchors], dim=-1)
        
        logits = self.fc(fused_features)
        return logits