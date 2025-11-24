import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
import copy
import json

# 모듈 임포트
from dataloader import MetaPortfolioDataset
from model_architecture import AssetUNet

class MetaTrainer:
    def __init__(self, k=4, tau=0.60, inner_lr=0.01, meta_lr=0.001, n_inner_steps=5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.k = k
        self.tau = tau
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.n_inner_steps = n_inner_steps
        
        # 데이터셋 로드
        self.train_ds = MetaPortfolioDataset(k=k, tau=tau, split='train')
        self.val_ds = MetaPortfolioDataset(k=k, tau=tau, split='val')
        
        # 모델 초기화
        self.model = AssetUNet(n_features=self.train_ds.n_features, n_assets=self.train_ds.n_assets).to(self.device)
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=meta_lr)
        self.criterion = nn.MSELoss()
        
        # 결과 저장 경로
        self.save_dir = f'./checkpoints/K{k}_tau{tau:.2f}'
        os.makedirs(self.save_dir, exist_ok=True)

    def inner_loop(self, model, support_x, support_y):
        """
        Task-specific adaptation (Inner Loop)
        """
        # 모델 복사 (Fast weights)
        # PyTorch의 higher 라이브러리를 쓰면 더 깔끔하지만, 
        # 여기서는 명시적으로 state_dict를 복사해서 사용 (First-Order MAML)
        fast_model = copy.deepcopy(model)
        fast_model.train()
        inner_opt = optim.SGD(fast_model.parameters(), lr=self.inner_lr)
        
        for _ in range(self.n_inner_steps):
            pred = fast_model(support_x)
            loss = self.criterion(pred, support_y)
            
            inner_opt.zero_grad()
            loss.backward()
            inner_opt.step()
            
        return fast_model

    def train_step(self, batch):
            """
            Meta-Training Step (Outer Loop) - FOMAML Implementation
            """
            meta_loss = 0
            batch_size = batch['support_x'].size(0)
            
            self.meta_optimizer.zero_grad()
            
            # [중요] 배치 내의 모든 Task에 대한 Gradient를 누적해야 함
            total_grads = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
            
            for i in range(batch_size):
                # 1. Task Data 준비
                sup_x = batch['support_x'][i].to(self.device)
                sup_y = batch['support_y'][i, -1, :].to(self.device)
                qry_x = batch['query_x'][i].to(self.device)
                qry_y = batch['query_y'][i, -1, :].to(self.device)
                
                # 2. Inner Loop (Adaptation)
                # fast_model은 self.model과 연결이 끊긴(Detached) 복사본
                fast_model = self.inner_loop(self.model, sup_x.unsqueeze(0), sup_y.unsqueeze(0))
                
                # 3. Query Loss 계산
                fast_model.eval()
                qry_pred = fast_model(qry_x.unsqueeze(0))
                loss = self.criterion(qry_pred, qry_y.unsqueeze(0))
                
                # 4. Gradient 계산 (w.r.t fast_model parameters)
                fast_model.zero_grad()
                loss.backward()
                
                # [핵심 수정] FOMAML: Fast Model의 Gradient를 가져와서 누적
                for n, p in fast_model.named_parameters():
                    if p.grad is not None:
                        total_grads[n] += p.grad.data / batch_size # Average over batch
                
                meta_loss += loss.item()
                
            # 5. Meta Update
            # 누적된 Gradient를 원본 모델(self.model)에 주입
            for n, p in self.model.named_parameters():
                if n in total_grads:
                    p.grad = total_grads[n]
            
            # Gradient Clipping & Step
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.meta_optimizer.step()
            
            return meta_loss / batch_size

    def validate(self):
        """Validation (Meta-Testing on Val Set)"""
        loader = DataLoader(self.val_ds, batch_size=4, shuffle=False)
        total_loss = 0
        count = 0
        
        for batch in loader:
            batch_size = batch['support_x'].size(0)
            for i in range(batch_size):
                sup_x = batch['support_x'][i].to(self.device)
                sup_y = batch['support_y'][i, -1, :].to(self.device)
                qry_x = batch['query_x'][i].to(self.device)
                qry_y = batch['query_y'][i, -1, :].to(self.device)
                
                # Adaptation
                fast_model = self.inner_loop(self.model, sup_x.unsqueeze(0), sup_y.unsqueeze(0))
                
                # Evaluation
                fast_model.eval()
                with torch.no_grad():
                    pred = fast_model(qry_x.unsqueeze(0))
                    loss = self.criterion(pred, qry_y.unsqueeze(0))
                    total_loss += loss.item()
                    count += 1
                    
        return total_loss / count

    def run_training(self, epochs=50):
        print(f"Starting Meta-Training (K={self.k}, Tau={self.tau})...")
        train_loader = DataLoader(self.train_ds, batch_size=4, shuffle=True)
        
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            steps = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                loss = self.train_step(batch)
                train_loss += loss
                steps += 1
                pbar.set_postfix({'loss': loss})
                
            avg_train_loss = train_loss / steps
            avg_val_loss = self.validate()
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            
            print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}")
            
            # Save Best Model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'best_meta_model.pth'))
                print("✓ Saved Best Model")
                
        # Save History
        with open(os.path.join(self.save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f)
            
        return history

if __name__ == "__main__":
    # Run Meta-Training
    trainer = MetaTrainer(k=4, tau=0.60, inner_lr=0.01, meta_lr=0.001, n_inner_steps=5)
    trainer.run_training(epochs=20) # 시간 관계상 20 Epoch만