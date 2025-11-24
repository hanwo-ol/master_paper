import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
from tqdm import tqdm

from dataloader import MetaPortfolioDataset
from model_architecture import AssetUNet

def run_diagnostics(k=4, tau=0.60):
    print("="*80)
    print("Meta-Learning Diagnostics: Adaptation Effect Analysis")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 데이터셋 (Test)
    ds = MetaPortfolioDataset(k=k, tau=tau, split='test')
    
    # 모델 로드
    meta_path = f'./checkpoints/K{k}_tau{tau:.2f}/best_meta_model.pth'
    model = AssetUNet(ds.n_features, ds.n_assets).to(device)
    model.load_state_dict(torch.load(meta_path, map_location=device))
    
    criterion = nn.MSELoss()
    
    results = {
        'pre_adapt_loss': [],
        'post_adapt_loss': [],
        'improvement': []
    }
    
    print("Analyzing Adaptation Effect on Test Episodes...")
    
    indices = np.random.choice(len(ds), min(100, len(ds)), replace=False)
    
    for idx in tqdm(indices):
        ep = ds[idx]
        
        # [수정] 차원 처리
        # ep['support_x']: (F, N, T) -> (1, F, N, T)
        sup_x = ep['support_x'].unsqueeze(0).to(device)
        
        # ep['support_y']: (T, N) -> 마지막 시점만 사용 -> (N,) -> (1, N)
        # 학습 때 Last Step Target을 썼으므로 여기서도 동일하게
        sup_y = ep['support_y'][-1, :].unsqueeze(0).to(device)
        
        qry_x = ep['query_x'].unsqueeze(0).to(device)
        qry_y = ep['query_y'][-1, :].unsqueeze(0).to(device)
        
        # 1. Pre-Adaptation Loss (theta_0)
        model.eval()
        with torch.no_grad():
            pred_0 = model(qry_x)
            loss_0 = criterion(pred_0, qry_y).item()
            
        # 2. Adaptation (Inner Loop)
        fast_model = copy.deepcopy(model)
        fast_model.train()
        optimizer = torch.optim.SGD(fast_model.parameters(), lr=0.001)
        
        for _ in range(5): # 5 steps
            p = fast_model(sup_x)
            l = criterion(p, sup_y)
            
            if torch.isnan(l): break
                
            optimizer.zero_grad()
            l.backward()
            # Clipping 추가
            torch.nn.utils.clip_grad_norm_(fast_model.parameters(), 1.0)
            optimizer.step()
            
        # 3. Post-Adaptation Loss (theta_K)
        fast_model.eval()
        with torch.no_grad():
            pred_k = fast_model(qry_x)
            loss_k = criterion(pred_k, qry_y).item()
            
        results['pre_adapt_loss'].append(loss_0)
        results['post_adapt_loss'].append(loss_k)
        
        # 개선율
        if loss_0 != 0:
            imp = (loss_0 - loss_k) / loss_0 * 100
        else:
            imp = 0.0
        results['improvement'].append(imp)
        
    # 통계 출력
    avg_imp = np.mean(results['improvement'])
    positive_ratio = np.mean(np.array(results['improvement']) > 0) * 100
    
    print(f"\n[Diagnostic Results]")
    print(f"Average Loss Improvement: {avg_imp:.2f}%")
    print(f"Positive Adaptation Ratio: {positive_ratio:.1f}%")
    
    # 시각화
    plt.figure(figsize=(10, 6))
    plt.hist(results['improvement'], bins=30, color='skyblue', edgecolor='black')
    plt.axvline(0, color='red', linestyle='--', label='Zero Improvement')
    plt.title('Distribution of Adaptation Improvement (Test Set)')
    plt.xlabel('Improvement (%)')
    plt.ylabel('Count')
    plt.legend()
    
    os.makedirs('./results', exist_ok=True)
    plt.savefig('./results/diagnosis_adaptation.png')
    print("✓ Plot saved to ./results/diagnosis_adaptation.png")
    
    if avg_imp > 0:
        print(">> DIAGNOSIS: Adaptation is working properly (Positive Transfer).")
    else:
        print(">> DIAGNOSIS: Negative Transfer detected! (Overfitting to Support Set).")

if __name__ == "__main__":
    run_diagnostics(k=4, tau=0.60)