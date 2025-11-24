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

from dataloader import MetaPortfolioDataset

# [핵심] Attention과 Skip Connection을 제거한 순수 CNN 모델 정의
class PureCNN(nn.Module):
    def __init__(self, n_features, n_assets):
        super().__init__()
        self.input_norm = nn.InstanceNorm2d(n_features, affine=False)
        
        # Receptive Field가 자산 축(Height)으로만 커지도록 설계
        # Kernel Size: (Asset=3, Time=3)
        self.conv1 = nn.Conv2d(n_features, 64, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d((2, 2)) # Asset도 줄이고 Time도 줄임
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d((2, 2))
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(256)
        
        # Global Average Pooling over Time, but keep Asset dimension?
        # No, we want to predict per asset.
        # U-Net이 아니므로, 여기서는 단순화를 위해 
        # "Local Pattern -> FC" 구조로 감.
        # 하지만 Asset별 예측을 위해 Conv1x1로 마무리하는 FCN 구조 채택
        
    def forward(self, x):
        # x: (B, F, N, T)
        x = self.input_norm(x)
        
        x = torch.relu(self.bn1(self.conv1(x)))
        # Pooling을 하면 Asset 차원(N)이 줄어드므로, 
        # 원본 N을 유지하려면 Padding을 잘 쓰거나 Upsampling 필요.
        # 여기서는 HRP 효과를 극대화하기 위해 Pooling 없이 Dilated Conv 사용이 나을 수 있으나,
        # 간단히 Padding=Same인 Conv만 쌓아서 테스트.
        
        return x

# 수정된 PureCNN (Pooling 없이 Local Conv만 쌓음)
class LocalCNN(nn.Module):
    def __init__(self, n_features, n_assets):
        super().__init__()
        self.input_norm = nn.InstanceNorm2d(n_features, affine=False)
        
        # Kernel (3, 1) -> 자산 축으로만 인접한 3개를 봄 (Time은 1)
        # HRP 효과를 보려면 자산 축의 Locality가 중요함
        self.conv1 = nn.Conv2d(n_features, 64, kernel_size=(5, 1), padding=(2, 0))
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(5, 1), padding=(2, 0))
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 64, kernel_size=(5, 1), padding=(2, 0))
        self.bn3 = nn.BatchNorm2d(64)
        
        self.final = nn.Conv2d(64, 1, kernel_size=1)
        
    def forward(self, x):
        x = self.input_norm(x)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        
        out = self.final(x) # (B, 1, N, T)
        return out[..., -1].squeeze(-1) # Last time step

def train_and_evaluate(order_indices, dataset, n_epochs=5, label=""):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    
    # LocalCNN 사용
    model = LocalCNN(n_features=dataset.n_features, n_assets=dataset.n_assets).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    losses = []
    print(f"Training [{label}]...")
    model.train()
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        count = 0
        for batch in loader:
            x = batch['support_x'].to(device)
            y = batch['support_y'][:, -1, :].to(device)
            
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

def run_local_cnn_experiment():
    print("="*80)
    print("Experiment 2E: Local CNN Ablation (Testing Spatial Inductive Bias)")
    print("="*80)
    
    dataset = MetaPortfolioDataset(k=4, tau=0.60, split='train')
    full_returns = dataset.Y.numpy()
    
    print("Computing Orders...")
    df_ret = pd.DataFrame(full_returns)
    corr = df_ret.corr().values
    dist = np.sqrt((1 - corr) / 2)
    dist = np.nan_to_num(dist)
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0)
    link_mat = linkage(squareform(dist, checks=False), method='ward')
    hrp_order = leaves_list(link_mat)
    
    np.random.seed(42)
    random_order = np.random.permutation(dataset.n_assets)
    
    n_epochs = 5
    hrp_losses = train_and_evaluate(hrp_order, dataset, n_epochs, "HRP Ordering")
    rand_losses = train_and_evaluate(random_order, dataset, n_epochs, "Random Ordering")
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_epochs+1), hrp_losses, 'b-o', label='HRP Ordering')
    plt.plot(range(1, n_epochs+1), rand_losses, 'r--s', label='Random Ordering')
    plt.title('Local CNN Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs('./results', exist_ok=True)
    plt.savefig('./results/exp2e_local_cnn.png')
    print("\n✓ Plot saved to ./results/exp2e_local_cnn.png")
    
    final_hrp = hrp_losses[-1]
    final_rand = rand_losses[-1]
    gap = (final_rand - final_hrp) / final_rand * 100
    
    print(f"\n[Final Loss - Local CNN]")
    print(f"HRP:    {final_hrp:.6f}")
    print(f"Random: {final_rand:.6f}")
    print(f"Improvement: {gap:.2f}%")

if __name__ == "__main__":
    run_local_cnn_experiment()