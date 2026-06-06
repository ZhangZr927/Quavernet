# --- START OF FILE stage2_config.py ---
import os
import torch

# ================= 路径配置 =================
DATA_DIR = "data"
MODELS_DIR = "stage2_models"
os.makedirs(MODELS_DIR, exist_ok=True)
STAGE1_MODEL_PATH = "init_models/best_rhythm_model.pth"

# ================= 数据参数 =================
SUBDIVISIONS = 48           
CHUNK_SIZE = 1024           
VALIDATION_SPLIT = 0.1

IN_CHANNELS = 80            
OUT_CLASSES = 4             

# ================= Stage 2 WGAN-GP 参数 =================
BATCH_SIZE = 32             
EPOCHS = 300                
CHECKPOINT_INTERVAL = 10    

LR_G = 2e-5                 
LR_D = 1e-4                 
BETA1 = 0.0                 
BETA2 = 0.9
N_CRITIC = 3                
LAMBDA_GP = 10.0            
LAMBDA_DRIFT = 0.001        

# 联合损失权重
LAMBDA_FOCAL = 2.0          
POS_WEIGHT = 10.0           

# 拒绝惩罚系数
LAMBDA_REJECT = 20.0        

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"