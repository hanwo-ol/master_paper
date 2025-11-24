import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
import json

from dataloader import MetaPortfolioDataset

# 1. LSTM Model
class AssetLSTM(nn.Module):
    def __init__(self, n_features, n_assets, hidden_size=128, num_layers=2):
        super().__init__()
        self.input_norm = nn.InstanceNorm1d(n_features, affine=False)
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x: (B, F, N, T) -> (B*N, T, F)
        B, F, N, T = x.size()
        x = x.permute(0, 2, 1, 3).reshape(B * N, F, T)
        x = self.input_norm(x)
        x = x.permute(0, 2, 1) 
        
        out, _ = self.lstm(x)
        out = out[:, -1, :] 
        pred = self.fc(out) 
        
        return pred.view(B, N)

# 2. Transformer Model
class AssetTransformer(nn.Module):
    def __init__(self, n_features, n_assets, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_norm = nn.InstanceNorm1d(n_features, affine=False)
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 60, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        
    def forward(self, x):
        B, F, N, T = x.size()
        x = x.permute(0, 2, 1, 3).reshape(B * N, F, T)
        x = self.input_norm(x)
        x = x.permute(0, 2, 1) 
        
        x = self.input_proj(x) + self.pos_encoder[:, :T, :]
        out = self.transformer(x)
        out = out.mean(dim=1) 
        pred = self.fc(out)
        
        return pred.view(B, N)

# Trainer
class BaselineTrainer:
    def __init__(self, model_type='lstm', k=4, tau=0.60, batch_size=16): # batch_size 인자 추가
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.batch_size = batch_size # 저장
        self.save_dir = f'./checkpoints/{model_type}_K{k}_tau{tau:.2f}'
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.train_ds = MetaPortfolioDataset(k=k, tau=tau, split='train')
        self.val_ds = MetaPortfolioDataset(k=k, tau=tau, split='val')
        
        if model_type == 'lstm':
            self.model = AssetLSTM(self.train_ds.n_features, self.train_ds.n_assets).to(self.device)
        elif model_type == 'transformer':
            self.model = AssetTransformer(self.train_ds.n_features, self.train_ds.n_assets).to(self.device)
            
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=1e-5)
        self.criterion = nn.MSELoss()
        
    def train(self, epochs=20, patience=5):
        print(f"Training {self.model_type.upper()} Baseline (Batch={self.batch_size})...")
        # self.batch_size 사용
        train_loader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                x = batch['support_x'].to(self.device)
                y = batch['support_y'][:, -1, :].to(self.device)
                
                self.optimizer.zero_grad()
                pred = self.model(x)
                loss = self.criterion(pred, y)
                
                if torch.isnan(loss): continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                train_loss += loss.item()
                
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch['query_x'].to(self.device)
                    y = batch['query_y'][:, -1, :].to(self.device)
                    pred = self.model(x)
                    loss = self.criterion(pred, y)
                    if not torch.isnan(loss):
                        val_loss += loss.item()
            
            avg_val = val_loss / len(val_loader)
            print(f"Val Loss: {avg_val:.6f}")
            
            # Model Selection & Early Stopping
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'best_model.pth'))
                patience_counter = 0 # Reset
                print("✓ Saved Best Model")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")
                
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
                
if __name__ == "__main__":
    import gc
    
    # LSTM (Batch 16)
    lstm_trainer = BaselineTrainer('lstm', batch_size=16)
    lstm_trainer.train(epochs=20, patience=5)
    
    del lstm_trainer
    gc.collect()
    torch.cuda.empty_cache()
    
    # Transformer (Batch 8로 감소) -> 안전하게 실행
    tf_trainer = BaselineTrainer('transformer', batch_size=8)
    tf_trainer.train(epochs=20, patience=5)