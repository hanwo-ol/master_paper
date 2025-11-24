import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os

# 모듈 임포트
from dataloader import MetaPortfolioDataset
from model_architecture import AssetUNet

def get_gradient_vector(model, loss):
    """모델의 모든 파라미터에 대한 Gradient를 1차원 벡터로 Flatten"""
    # 기존 Gradient 초기화
    model.zero_grad()
    # Backward 수행
    loss.backward()
    
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            # NaN이나 Inf가 있는지 체크하여 0으로 치환 (Safety)
            g = param.grad.view(-1)
            if torch.isnan(g).any() or torch.isinf(g).any():
                g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
            grads.append(g)
            
    if not grads:
        return torch.tensor([])
        
    return torch.cat(grads)

def run_gradient_alignment_experiment():
    print("="*80)
    print("Experiment 1: Cross-Regime Gradient Alignment Analysis (Robust Ver.)")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. 데이터 로드
    dataset = MetaPortfolioDataset(k=4, tau=0.60, split='train')
    # Batch size 1은 너무 불안정할 수 있으므로 4로 증가시켜 평균적인 Gradient를 봅니다.
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 2. 모델 초기화 및 Pre-training (Warm-up)
    print("\n[Phase 1] Pre-training model for 1 epoch (Warm-up)...")
    model = AssetUNet(n_features=dataset.n_features, n_assets=dataset.n_assets).to(device)
    
    # LR을 조금 낮춤 (안정성 확보)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    model.train()
    
    # Warm-up Loop
    for i, batch in enumerate(tqdm(loader, total=50, desc="Warm-up")):
        if i >= 50: break # 50 step만
        
        x = batch['support_x'].to(device)
        y_true = batch['support_y'][:, -1, :].to(device)
        
        y_pred = model(x)
        loss = criterion(y_pred, y_true)
        
        if torch.isnan(loss):
            print("Warning: NaN loss during warm-up. Skipping batch.")
            continue
            
        optimizer.zero_grad()
        loss.backward()
        # Gradient Clipping 추가
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
    print("✓ Warm-up complete.")
    
    # 3. Regime별 Gradient 수집
    print("\n[Phase 2] Collecting Gradients per Regime...")
    regime_grads = {0: [], 1: [], 2: [], 3: []}
    samples_per_regime = 90 # 샘플 수 조정
    
    model.eval()
    
    # Loader 재설정 (Batch size 1로 개별 샘플 Gradient 확인)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    pbar = tqdm(total=samples_per_regime * 4, desc="Collecting")
    
    for batch in loader:
        regime = batch['regime'].item()
        
        if len(regime_grads[regime]) >= samples_per_regime:
            continue
            
        x = batch['support_x'].to(device)
        y_true = batch['support_y'][:, -1, :].to(device)
        
        # Forward
        y_pred = model(x)
        loss = criterion(y_pred, y_true)
        
        if torch.isnan(loss):
            continue

        # Get Gradient
        grad_vec = get_gradient_vector(model, loss).cpu()
        
        # 유효성 검사: Norm이 0이거나 NaN이면 스킵
        if grad_vec.numel() == 0 or torch.isnan(grad_vec).any() or grad_vec.norm() < 1e-6:
            continue
            
        regime_grads[regime].append(grad_vec)
        pbar.update(1)
        
        if all(len(v) >= samples_per_regime for v in regime_grads.values()):
            break
            
    pbar.close()
    
    # 수집된 샘플 수 확인
    print("\nCollected samples per regime:")
    for r, grads in regime_grads.items():
        print(f"  Regime {r}: {len(grads)} samples")
    
    # 4. Cosine Similarity 계산
    print("\n[Phase 3] Computing Cosine Similarity Matrix...")
    n_regimes = 4
    sim_matrix = np.zeros((n_regimes, n_regimes))
    
    for i in range(n_regimes):
        for j in range(n_regimes):
            if len(regime_grads[i]) == 0 or len(regime_grads[j]) == 0:
                sim_matrix[i, j] = np.nan
                continue
                
            grads_i = torch.stack(regime_grads[i]) # (N, D)
            grads_j = torch.stack(regime_grads[j]) # (N, D)
            
            # Normalize with epsilon to avoid division by zero
            grads_i = torch.nn.functional.normalize(grads_i, p=2, dim=1, eps=1e-8)
            grads_j = torch.nn.functional.normalize(grads_j, p=2, dim=1, eps=1e-8)
            
            # Cosine Similarity
            sims = torch.mm(grads_i, grads_j.t())
            avg_sim = sims.mean().item()
            
            sim_matrix[i, j] = avg_sim
            
    print("\nGradient Correlation Matrix:")
    print(np.round(sim_matrix, 4))
    
    # 5. 시각화
    if not np.isnan(sim_matrix).all():
        plt.figure(figsize=(8, 6))
        sns.heatmap(sim_matrix, annot=True, cmap='coolwarm', vmin=-0.2, vmax=1.0, fmt='.3f')
        plt.title('Cross-Regime Gradient Alignment (Cosine Similarity)')
        plt.xlabel('Regime Task')
        plt.ylabel('Regime Task')
        
        os.makedirs('./results', exist_ok=True)
        plt.savefig('./results/exp1_gradient_alignment.png')
        print("\n✓ Plot saved to ./results/exp1_gradient_alignment.png")
        
        # 해석
        diag_vals = np.diag(sim_matrix)
        off_diag_vals = sim_matrix[~np.eye(sim_matrix.shape[0], dtype=bool)]
        
        diag_mean = np.nanmean(diag_vals)
        off_diag_mean = np.nanmean(off_diag_vals)
        
        print(f"\n[Interpretation]")
        print(f"Average Intra-Regime Similarity (Diagonal): {diag_mean:.4f}")
        print(f"Average Inter-Regime Similarity (Off-Diagonal): {off_diag_mean:.4f}")
        
        if off_diag_mean < diag_mean * 0.5:
             print(">> RESULT: Low cross-regime alignment confirmed. Meta-learning justified.")
        else:
             print(">> RESULT: Moderate/High alignment. Tasks might be similar.")
    else:
        print(">> ERROR: Matrix contains all NaNs. Check data scaling or model initialization.")

if __name__ == "__main__":
    run_gradient_alignment_experiment()