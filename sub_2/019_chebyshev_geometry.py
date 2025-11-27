import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import copy
from tqdm import tqdm

from dataloader import MetaPortfolioDataset
from model_architecture import AssetUNet

def get_flat_params(model):
    """모델 파라미터를 1차원 벡터로 Flatten"""
    params = []
    for param in model.parameters():
        params.append(param.view(-1))
    return torch.cat(params).detach().cpu().numpy()

def train_regime_expert(k, tau, regime_idx, epochs=5):
    """특정 Regime 데이터로만 학습하여 Regime Optima theta_s* 찾기"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 해당 Regime 데이터만 로드 (Train Set)
    ds = MetaPortfolioDataset(k=k, tau=tau, split='train')
    # Filter by regime
    regime_indices = [i for i, ep in enumerate(ds.episodes) if ep['support_regime'] == regime_idx]
    
    if not regime_indices:
        print(f"Warning: No data for regime {regime_idx}")
        return None
        
    subset = torch.utils.data.Subset(ds, regime_indices)
    loader = DataLoader(subset, batch_size=16, shuffle=True)
    
    model = AssetUNet(ds.n_features, ds.n_assets).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print(f"Training Expert for Regime {regime_idx} ({len(subset)} episodes)...")
    model.train()
    for _ in range(epochs):
        for batch in loader:
            x = batch['support_x'].to(device)
            y = batch['support_y'][:, -1, :].to(device)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
    return get_flat_params(model)

def run_chebyshev_geometry(k=4, tau=0.60):
    print("="*80)
    print("Experiment B.2: Chebyshev-Center Geometry Visualization")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Meta-Initialization theta_0 로드
    meta_path = f'./checkpoints/K{k}_tau{tau:.2f}/best_meta_model.pth'
    meta_model = AssetUNet(38, 501).to(device) # Feature/Asset 수 하드코딩 주의 (Loader에서 가져오는게 안전)
    # Loader를 잠깐 만들어서 정보 가져오기
    temp_ds = MetaPortfolioDataset(k=k, tau=tau, split='train')
    meta_model = AssetUNet(temp_ds.n_features, temp_ds.n_assets).to(device)
    meta_model.load_state_dict(torch.load(meta_path, map_location=device))
    
    theta_0 = get_flat_params(meta_model)
    
    # 2. Random Initialization theta_rand
    rand_model = AssetUNet(temp_ds.n_features, temp_ds.n_assets).to(device)
    theta_rand = get_flat_params(rand_model)
    
    # 3. Regime Optima theta_s* 학습
    theta_stars = []
    regime_labels = []
    
    for r in range(k):
        theta_s = train_regime_expert(k, tau, r)
        if theta_s is not None:
            theta_stars.append(theta_s)
            regime_labels.append(f"Regime {r} Opt")
            
    # 4. 거리 분석 (Distance Analysis)
    print("\n[Distance Analysis]")
    dists_meta = [np.linalg.norm(theta_0 - t_s) for t_s in theta_stars]
    dists_rand = [np.linalg.norm(theta_rand - t_s) for t_s in theta_stars]
    
    print(f"Meta-Init Distances: {np.round(dists_meta, 4)}")
    print(f"Random-Init Distances: {np.round(dists_rand, 4)}")
    print(f"Meta Max Dist: {np.max(dists_meta):.4f}")
    print(f"Random Max Dist: {np.max(dists_rand):.4f}")
    
    if np.max(dists_meta) < np.max(dists_rand):
        print(">> RESULT: Meta-Init is closer to the worst-case regime (Chebyshev Property).")
    
    # 5. PCA Visualization
    # 모든 벡터를 모아서 PCA 수행
    all_vecs = np.vstack([theta_0, theta_rand] + theta_stars)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(all_vecs)
    
    plt.figure(figsize=(10, 8))
    
    # Plot Regime Optima
    for i, label in enumerate(regime_labels):
        plt.scatter(coords[2+i, 0], coords[2+i, 1], s=200, label=label, marker='*')
        
    # Plot Meta-Init
    plt.scatter(coords[0, 0], coords[0, 1], s=300, c='red', label='Meta-Init (theta_0)', marker='X')
    
    # Plot Random-Init
    plt.scatter(coords[1, 0], coords[1, 1], s=100, c='gray', label='Random-Init', marker='o', alpha=0.5)
    
    # Draw connections from Meta to Regimes
    for i in range(len(theta_stars)):
        plt.plot([coords[0, 0], coords[2+i, 0]], [coords[0, 1], coords[2+i, 1]], 'r--', alpha=0.3)
        
    plt.title('Geometry of Meta-Initialization vs Regime Optima (PCA)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs('./results', exist_ok=True)
    plt.savefig('./results/exp_b2_chebyshev_geometry.png')
    print("\n✓ Plot saved to ./results/exp_b2_chebyshev_geometry.png")

if __name__ == "__main__":
    run_chebyshev_geometry(k=4, tau=0.60)