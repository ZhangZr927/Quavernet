# --- START OF FILE stage2_train.py ---
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from stage2_config import *
from dataset import get_dataloaders 
from stage2_model import MaskedChartGenerator, ConditionalDiscriminator, RhythmNet

class WeightedFocalLoss(nn.Module):
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

def compute_gradient_penalty(D, real_samples, fake_samples, diff, audio_cond):
    alpha = torch.rand(real_samples.size(0), 1, 1).to(DEVICE)
    alpha = alpha.expand(-1, real_samples.size(1), real_samples.size(2))
    
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(audio_cond, interpolates, diff)
        
    fake = torch.ones(d_interpolates.size()).to(DEVICE)
    gradients = torch.autograd.grad(
        outputs=d_interpolates, inputs=interpolates,
        grad_outputs=fake, create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    
    gradients = gradients.reshape(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

def generate_gpu_mask(rhythm_model, X, diff):
    """在一阶段特征图上快速进行 NMS，提取极其尖锐的一维掩码"""
    with torch.no_grad():
        logits_1d = rhythm_model(X, diff)  # (B, L, 1)
        probs = torch.sigmoid(logits_1d).permute(0, 2, 1) # 变成 (B, 1, L)
        
        # 局部极大值筛选
        local_max = F.max_pool1d(probs, kernel_size=5, stride=1, padding=2)
        mask = ((probs == local_max) & (probs > 0.3)).float()
    return mask  # (B, 1, L)

def main():
    print(f"[*] Using device: {DEVICE}")
    train_loader, val_loader = get_dataloaders()
    
    # --- 加载一阶段模型 ---
    print("[*] Loading pre-trained Stage 1 Rhythm Model...")
    rhythm_model = RhythmNet().to(DEVICE)
    rhythm_model.load_state_dict(torch.load(STAGE1_MODEL_PATH, map_location=DEVICE))
    rhythm_model.eval() # 永远不训练一阶段
    
    # --- 初始化二阶段模型 ---
    G = MaskedChartGenerator().to(DEVICE)
    D = ConditionalDiscriminator().to(DEVICE)
    
    opt_G = torch.optim.Adam(G.parameters(), lr=LR_G, betas=(BETA1, BETA2))
    opt_D = torch.optim.Adam(D.parameters(), lr=LR_D, betas=(BETA1, BETA2))
    focal_criterion = WeightedFocalLoss(pos_weight=POS_WEIGHT)

    for epoch in range(1, EPOCHS + 1):
        G.train(); D.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        
        for batch_idx, (X, Y_4d, diff) in enumerate(pbar):
            X, Y_4d, diff = X.to(DEVICE), Y_4d.to(DEVICE), diff.to(DEVICE)
            
            # 1. 在线获取节奏限制区 Mask
            mask_1d = generate_gpu_mask(rhythm_model, X, diff) 
            mask_aligned = mask_1d.permute(0, 2, 1) # (B, L, 1)
            
            # 2. 对齐真谱 (极其重要)：屏蔽掉所有被一阶段漏掉的按键，
            #    保证判别器和FocalLoss不会因为这些漏键而冤枉二阶段模型
            real_chart_masked = Y_4d * mask_aligned
            
            # =================== 3. 训练鉴别器 D ===================
            opt_D.zero_grad()
            with torch.no_grad():
                logits_fake = G(X, diff, mask_1d)
                fake_chart = torch.sigmoid(logits_fake)
            
            score_real = D(X, real_chart_masked, diff).mean()
            score_fake = D(X, fake_chart, diff).mean()
            
            gp = compute_gradient_penalty(D, real_chart_masked, fake_chart, diff, audio_cond=X)
            drift_penalty = LAMBDA_DRIFT * (score_real ** 2 + score_fake ** 2)
            
            loss_D = score_fake - score_real + LAMBDA_GP * gp + drift_penalty
            loss_D.backward()
            opt_D.step()
            w_dist = (score_real - score_fake).item()
            
            # =================== 4. 训练生成器 G ===================
            loss_G_focal, loss_reject, r_ratio = torch.tensor(0.0), torch.tensor(0.0), 0.0
            
            if batch_idx % N_CRITIC == 0:
                opt_G.zero_grad()
                
                logits_fake_for_G = G(X, diff, mask_1d)
                fake_chart_for_G = torch.sigmoid(logits_fake_for_G)
                
                loss_G_wgan = -D(X, fake_chart_for_G, diff).mean()
                
                # 【惩罚 1】保底 Focal Loss (只针对被选中的点)
                loss_G_focal = focal_criterion(logits_fake_for_G, real_chart_masked)
                
                # 【惩罚 2】未放键惩罚 (Rejection Penalty)
                lane_sum = fake_chart_for_G.sum(dim=-1, keepdim=True) # (B, L, 1)
                rejection = F.relu(1.0 - lane_sum) 
                
                total_mask_points = mask_aligned.sum() + 1e-5
                r_ratio = (rejection * mask_aligned).sum() / total_mask_points
                loss_reject = LAMBDA_REJECT * r_ratio
                
                loss_G = loss_G_wgan + LAMBDA_FOCAL * loss_G_focal + loss_reject
                loss_G.backward()
                opt_G.step()
            
            pbar.set_postfix({
                'D_Real': f"{score_real:.2f}",
                'D_Fake': f"{score_fake:.2f}",
                'W_Dist': f"{w_dist:.2f}",
                'Rej_%': f"{r_ratio*100:.1f}%",
                'L_Rej': f"{loss_reject.item():.2f}",
                'L_G': f"{loss_G.item():.2f}"
            })
        
        if epoch % CHECKPOINT_INTERVAL == 0 or epoch == EPOCHS:
            save_path = os.path.join(MODELS_DIR, f"stage2_generator_ep{epoch}.pth")
            torch.save(G.state_dict(), save_path)
            torch.save(G.state_dict(), os.path.join(MODELS_DIR, "best_stage2_generator.pth"))

if __name__ == "__main__":
    main()