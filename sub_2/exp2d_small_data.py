import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

# 모듈 임포트
from dataloader import MetaPortfolioDataset
from model_architecture import AssetUNet

def train_and_evaluate(order_indices, dataset, n_epochs=10, label=""):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Seed 고정
    torch.manual_seed(42)
    model = AssetUNet(n_features=dataset.n_features, n_assets=dataset.n_assets).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # [핵심] 데이터의 10%만 사용 (Small Data Regime)
    # 전체 약 2500개 -> 250개만 사용
    subset_size = int(len(dataset) * 0.05)
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42))[:subset_size]
    subset = Subset(dataset, indices)
    
    loader = DataLoader(subset, batch_size=16, shuffle=True)
    
    losses = []
    print(f"Training [{label}] with {subset_size} episodes...")
    
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0
        count = 0
        
        for batch in loader:
            x = batch['support_x'].to(device)
            y = batch['support_y'][:, -1, :].to(device)
            
            # 자산 순서 재배열
            x_ordered = x[:, :, order_indices, :]
            y_ordered = y[:, order_indices]
            
            optimizer.zero_grad()
            pred = model(x_ordered)
            loss = criterion(pred, y_ordered)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            count += 1
            
        avg_loss = epoch_loss / count
        losses.append(avg_loss)
        
    return losses

def run_small_data_experiment():
    print("="*80)
    print("Experiment 2D: Small Data Efficiency (Few-Shot Regime)")
    print("="*80)
    
    dataset = MetaPortfolioDataset(k=4, tau=0.60, split='train')
    full_returns = dataset.Y.numpy()
    
    # HRP Order 계산
    print("Computing Orders...")
    df_ret = pd.DataFrame(full_returns)
    corr = df_ret.corr().values
    dist = np.sqrt((1 - corr) / 2)
    dist = np.nan_to_num(dist)
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0)
    link_mat = linkage(squareform(dist, checks=False), method='ward')
    hrp_order = leaves_list(link_mat)
    
    # Random Order
    np.random.seed(42)
    random_order = np.random.permutation(dataset.n_assets)
    
    # 학습 (Epoch을 좀 더 늘려서 수렴 속도 차이 확인)
    n_epochs = 15
    hrp_losses = train_and_evaluate(hrp_order, dataset, n_epochs, "HRP Ordering")
    rand_losses = train_and_evaluate(random_order, dataset, n_epochs, "Random Ordering")
    
    # 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_epochs+1), hrp_losses, 'b-o', label='HRP Ordering')
    plt.plot(range(1, n_epochs+1), rand_losses, 'r--s', label='Random Ordering')
    
    plt.title('Small Data Learning Curve (10% Data)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs('./results', exist_ok=True)
    plt.savefig('./results/exp2d_small_data.png')
    print("\n✓ Plot saved to ./results/exp2d_small_data.png")
    
    final_hrp = hrp_losses[-1]
    final_rand = rand_losses[-1]
    gap = (final_rand - final_hrp) / final_rand * 100
    
    print(f"\n[Final Loss - Small Data]")
    print(f"HRP:    {final_hrp:.6f}")
    print(f"Random: {final_rand:.6f}")
    print(f"Improvement: {gap:.2f}%")

if __name__ == "__main__":
    run_small_data_experiment()