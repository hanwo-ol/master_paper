import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
import copy
import json

from dataloader import MetaPortfolioDataset
from model_architecture import AssetUNet

class MetaTrainer:
    def __init__(self, k=4, tau=0.60, inner_lr=0.001, meta_lr=0.0001, n_inner_steps=5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.k = k
        self.tau = tau
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.n_inner_steps = n_inner_steps
        
        self.train_ds = MetaPortfolioDataset(k=k, tau=tau, split='train')
        self.val_ds = MetaPortfolioDataset(k=k, tau=tau, split='val')
        
        self.model = AssetUNet(n_features=self.train_ds.n_features, n_assets=self.train_ds.n_assets).to(self.device)
        
        # [수정] Weight Decay 추가 (Regularization)
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=meta_lr, weight_decay=1e-5)
        
        # [수정] Learning Rate Scheduler 추가
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.meta_optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        
        self.criterion = nn.MSELoss()
        self.save_dir = f'./checkpoints/K{k}_tau{tau:.2f}'
        os.makedirs(self.save_dir, exist_ok=True)

    def inner_loop(self, model, support_x, support_y):
        fast_model = copy.deepcopy(model)
        fast_model.train()
        inner_opt = optim.SGD(fast_model.parameters(), lr=self.inner_lr)
        
        for _ in range(self.n_inner_steps):
            pred = fast_model(support_x)
            loss = self.criterion(pred, support_y)
            
            if torch.isnan(loss): break
                
            inner_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fast_model.parameters(), 1.0)
            inner_opt.step()
            
        return fast_model

    def train_step(self, batch):
        meta_loss = 0
        batch_size = batch['support_x'].size(0)
        self.meta_optimizer.zero_grad()
        
        total_grads = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        valid_tasks = 0
        
        for i in range(batch_size):
            sup_x = batch['support_x'][i].to(self.device)
            sup_y = batch['support_y'][i, -1, :].to(self.device)
            qry_x = batch['query_x'][i].to(self.device)
            qry_y = batch['query_y'][i, -1, :].to(self.device)
            
            # Inner Loop
            fast_model = self.inner_loop(self.model, sup_x.unsqueeze(0), sup_y.unsqueeze(0))
            
            # Query Loss
            fast_model.eval()
            qry_pred = fast_model(qry_x.unsqueeze(0))
            loss = self.criterion(qry_pred, qry_y.unsqueeze(0))
            
            if torch.isnan(loss): continue
                
            # FOMAML Gradient Accumulation
            fast_model.zero_grad()
            loss.backward()
            
            for n, p in fast_model.named_parameters():
                if p.grad is not None:
                    if not torch.isnan(p.grad).any() and not torch.isinf(p.grad).any():
                        total_grads[n] += p.grad.data
            
            meta_loss += loss.item()
            valid_tasks += 1
            
        if valid_tasks == 0: return 0.0
            
        for n, p in self.model.named_parameters():
            if n in total_grads:
                p.grad = total_grads[n] / valid_tasks
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.meta_optimizer.step()
        
        return meta_loss / valid_tasks

    def validate(self):
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
                
                fast_model = self.inner_loop(self.model, sup_x.unsqueeze(0), sup_y.unsqueeze(0))
                
                fast_model.eval()
                with torch.no_grad():
                    pred = fast_model(qry_x.unsqueeze(0))
                    loss = self.criterion(pred, qry_y.unsqueeze(0))
                    if not torch.isnan(loss):
                        total_loss += loss.item()
                        count += 1
        
        if count == 0: return float('inf')
        return total_loss / count

    def run_training(self, epochs=20):
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
                pbar.set_postfix({'loss': f"{loss:.6f}"})
                
            avg_train_loss = train_loss / steps if steps > 0 else 0
            avg_val_loss = self.validate()
            
            # Scheduler Step
            self.scheduler.step(avg_val_loss)
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            
            print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'best_meta_model.pth'))
                print("✓ Saved Best Model")
                
        with open(os.path.join(self.save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f)
            
        return history

if __name__ == "__main__":
    trainer = MetaTrainer(k=4, tau=0.60, inner_lr=0.001, meta_lr=0.0001, n_inner_steps=5)
    trainer.run_training(epochs=20)