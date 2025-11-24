import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
import json

from dataloader import MetaPortfolioDataset
from model_architecture import AssetUNet

class PooledTrainer:
    def __init__(self, k=4, tau=0.60, lr=0.0001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.k = k
        self.tau = tau
        
        # 데이터셋 로드
        self.train_ds = MetaPortfolioDataset(k=k, tau=tau, split='train')
        self.val_ds = MetaPortfolioDataset(k=k, tau=tau, split='val')
        
        # 모델 초기화
        self.model = AssetUNet(n_features=self.train_ds.n_features, n_assets=self.train_ds.n_assets).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3)
        self.criterion = nn.MSELoss()
        
        self.save_dir = f'./checkpoints/pooled_K{k}_tau{tau:.2f}'
        os.makedirs(self.save_dir, exist_ok=True)

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        steps = 0
        
        for batch in tqdm(loader, desc="Training"):
            # Pooled Training은 Support/Query 구분 없이 Support 데이터를 학습에 사용
            # (Meta-Learning과 데이터 양을 맞추기 위해 Support Set만 사용하거나 둘 다 사용 가능)
            # 여기서는 Support Set을 Input, 마지막 시점을 Target으로 사용
            x = batch['support_x'].to(self.device)
            y = batch['support_y'][:, -1, :].to(self.device)
            
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.criterion(pred, y)
            
            if torch.isnan(loss): continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            steps += 1
            
        return total_loss / steps if steps > 0 else 0

    def validate(self, loader):
        self.model.eval()
        total_loss = 0
        steps = 0
        
        with torch.no_grad():
            for batch in loader:
                x = batch['query_x'].to(self.device) # Val에서는 Query Set으로 평가 (Fair Comparison)
                y = batch['query_y'][:, -1, :].to(self.device)
                
                pred = self.model(x)
                loss = self.criterion(pred, y)
                
                if not torch.isnan(loss):
                    total_loss += loss.item()
                    steps += 1
                    
        return total_loss / steps if steps > 0 else float('inf')

    def run_training(self, epochs=20):
        print(f"Starting Pooled Training (Baseline)...")
        train_loader = DataLoader(self.train_ds, batch_size=16, shuffle=True) # 배치 사이즈 키움
        val_loader = DataLoader(self.val_ds, batch_size=16, shuffle=False)
        
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}, LR={current_lr:.2e}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'best_pooled_model.pth'))
                print("✓ Saved Best Pooled Model")
                
        with open(os.path.join(self.save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f)

if __name__ == "__main__":
    trainer = PooledTrainer(k=4, tau=0.60, lr=0.0001)
    trainer.run_training(epochs=20)