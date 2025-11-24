import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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

def get_hrp_order(df):
    """HRP 순서 인덱스 반환"""
    returns_df = df.pivot(index='date', columns='ticker', values='returns').fillna(0)
    corr = returns_df.corr().values
    dist = np.sqrt((1 - corr) / 2)
    dist = np.nan_to_num(dist)
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0)
    condensed_dist = squareform(dist, checks=False)
    link_mat = linkage(condensed_dist, method='ward')
    return leaves_list(link_mat)

def train_and_evaluate(order_indices, dataset, n_epochs=5, label=""):
    """주어진 자산 순서로 모델 학습"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 데이터셋의 자산 순서를 재배열하기 위한 매핑
    # dataset.X shape: (Time, Asset, Feature)
    # 우리는 모델 입력 전에 순서를 바꿀 것임
    
    # 모델 초기화 (매번 동일한 초기화)
    torch.manual_seed(42)
    model = AssetUNet(n_features=dataset.n_features, n_assets=dataset.n_assets).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    losses = []
    
    print(f"Training [{label}]...")
    model.train()
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        count = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False)
        for batch in pbar:
            # 원본 데이터: (B, F, N, T)
            x = batch['support_x'].to(device)
            y = batch['support_y'][:, -1, :].to(device) # (B, N)
            
            # 자산 순서 재배열 (Asset Dimension = 2)
            # x: (B, F, Asset, T)
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
        print(f"  Epoch {epoch+1}: Loss = {avg_loss:.6f}")
        
    return losses

def run_learning_efficiency_experiment():
    print("="*80)
    print("Experiment 2B: HRP vs Random Ordering Learning Efficiency")
    print("="*80)
    
    # 1. 데이터 로드
    # HRP 순서를 계산하기 위해 Raw Panel도 필요하지만, 
    # Dataset 내부의 X 텐서를 이용해 상관계수를 구하는 게 더 정확함
    dataset = MetaPortfolioDataset(k=4, tau=0.60, split='train')
    
    # 전체 기간 Returns 추출 (Time, Asset)
    full_returns = dataset.Y.numpy()
    
    # 2. 순서 결정
    print("Computing Orders...")
    
    # HRP Order
    # Correlation 계산
    df_ret = pd.DataFrame(full_returns)
    corr = df_ret.corr().values
    dist = np.sqrt((1 - corr) / 2)
    dist = np.nan_to_num(dist)
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0)
    condensed_dist = squareform(dist, checks=False)
    link_mat = linkage(condensed_dist, method='ward')
    hrp_order = leaves_list(link_mat)
    
    # Random Order (Seed 고정)
    np.random.seed(42)
    random_order = np.random.permutation(dataset.n_assets)
    
    # 3. 비교 학습
    n_epochs = 5
    
    hrp_losses = train_and_evaluate(hrp_order, dataset, n_epochs, "HRP Ordering")
    rand_losses = train_and_evaluate(random_order, dataset, n_epochs, "Random Ordering")
    
    # 4. 결과 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_epochs+1), hrp_losses, 'b-o', label='HRP Ordering', linewidth=2)
    plt.plot(range(1, n_epochs+1), rand_losses, 'r--s', label='Random Ordering', linewidth=2)
    
    plt.title('Learning Curve Comparison: HRP vs Random')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs('./results', exist_ok=True)
    plt.savefig('./results/exp2b_learning_efficiency.png')
    print("\n✓ Plot saved to ./results/exp2b_learning_efficiency.png")
    
    # 결과 해석
    final_hrp = hrp_losses[-1]
    final_rand = rand_losses[-1]
    gap = (final_rand - final_hrp) / final_rand * 100
    
    print(f"\n[Final Loss]")
    print(f"HRP:    {final_hrp:.6f}")
    print(f"Random: {final_rand:.6f}")
    print(f"Improvement: {gap:.2f}%")
    
    if final_hrp < final_rand:
        print(">> RESULT: HRP Ordering leads to better/faster learning. Structural bias confirmed.")
    else:
        print(">> RESULT: No advantage observed.")

if __name__ == "__main__":
    run_learning_efficiency_experiment()