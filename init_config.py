# --- START OF FILE init_config.py ---
import os
import torch

# ================= 路径配置 =================
DATA_DIR = "data"
MODELS_DIR = "init_models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ================= 数据与切割参数 (严格对齐主模型) =================
SUBDIVISIONS = 48           
CHUNK_SIZE = 1024           
VALIDATION_SPLIT = 0.1

# ================= CNN 参数 =================
IN_CHANNELS = 80            
CNN_CHANNELS = 128           
CNN_KERNEL_SIZE = 5         
RES_BLOCKS = 3

# ================= RNN & Attention 参数 =================
RNN_HIDDEN_SIZE = 256       
RNN_LAYERS = 2              
DROPOUT = 0.1               
ATTENTION_HEADS = 8

# ================= 输出与训练参数 =================
OUT_CLASSES = 1 
BATCH_SIZE = 32             
LEARNING_RATE = 5e-5       
EPOCHS = 400                
POS_WEIGHT = 5.0
CHECKPOINT_INTERVAL = 25    

# 硬件配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"