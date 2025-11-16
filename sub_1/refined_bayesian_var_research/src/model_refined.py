# ============================================================================
# model_refined.py (NLL LOSS FIXED)
# Stage 3: Bayesian NN with Calibration Loss
# FIX: Correct Gaussian NLL formula + Proper scaling
# ============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class BayesianVaRNN(nn.Module):
    """MC Dropout을 이용한 Bayesian VaR 추정 신경망"""
    
    def __init__(self, input_dim: int = 11, hidden_dim: int = 128, 
                 dropout_rate: float = 0.2):
        super(BayesianVaRNN, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # VaR 점 추정
        self.var_mean = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
            # Softplus 제거 (VaR는 음수 가능)
        )
        
        # Aleatoric uncertainty (log scale로 예측)
        self.aleatoric_log_std = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
            # Raw output → exp로 변환
        )
        
        # CVaR
        self.cvar_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.dropout_rate = dropout_rate
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with log_std → std conversion"""
        features = self.encoder(x)
        var_mean = self.var_mean(features)
        
        # FIX: log_std로 예측 후 exp
        log_std = self.aleatoric_log_std(features)
        aleatoric_std = torch.exp(log_std)
        
        # Clip to reasonable range
        aleatoric_std = torch.clamp(aleatoric_std, min=1e-6, max=1.0)
        
        cvar_pred = self.cvar_head(features)
        
        return var_mean, aleatoric_std, cvar_pred
    
    def mc_dropout_forward(self, x: torch.Tensor, n_samples: int = 100) -> torch.Tensor:
        """MC Dropout inference"""
        self.train()
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                var_mean, _, _ = self.forward(x)
                predictions.append(var_mean)
        
        predictions = torch.cat(predictions, dim=1)
        return predictions


class BayesianVaRLoss(nn.Module):
    """
    수정된 Loss 함수 (올바른 Gaussian NLL)
    
    수식:
    L_NLL = 0.5 * log(2π) + log(σ) + 0.5 * [(y - μ) / σ]²
    
    간소화 (상수 제거):
    L_NLL = log(σ) + 0.5 * [(y - μ) / σ]²
    """
    
    def __init__(self, lambda_cal: float = 1.0, lambda_cvar: float = 0.1):
        super(BayesianVaRLoss, self).__init__()
        self.lambda_cal = lambda_cal
        self.lambda_cvar = lambda_cvar
    
    def forward(self, var_pred: torch.Tensor, aleatoric_std: torch.Tensor, 
                y: torch.Tensor, cvar_pred: torch.Tensor = None,
                confidence: float = 0.95) -> Dict[str, torch.Tensor]:
        """
        올바른 Gaussian NLL 계산
        """
        var_pred = var_pred.squeeze()
        aleatoric_std = aleatoric_std.squeeze()
        y = y.squeeze()
        
        eps = 1e-6
        aleatoric_std = torch.clamp(aleatoric_std, min=eps, max=10.0)
        
        # ===================================================================
        # FIX: 올바른 Gaussian Negative Log-Likelihood
        # ===================================================================
        # L_NLL = log(σ) + 0.5 * [(y - μ) / σ]²
        
        log_std = torch.log(aleatoric_std)
        squared_error = ((y - var_pred) / aleatoric_std) ** 2
        
        nll_loss = torch.mean(log_std + 0.5 * squared_error)
        
        # ===================================================================
        # Calibration Loss (unchanged)
        # ===================================================================
        z_score = 1.96 if confidence == 0.95 else 2.576
        
        lower_bound = var_pred - z_score * aleatoric_std
        upper_bound = var_pred + z_score * aleatoric_std
        
        in_interval = ((y >= lower_bound) & (y <= upper_bound)).float()
        actual_coverage = in_interval.mean()
        target_coverage = torch.tensor(confidence, device=var_pred.device)
        
        calibration_loss = (actual_coverage - target_coverage) ** 2
        
        # ===================================================================
        # CVaR Loss (unchanged)
        # ===================================================================
        cvar_loss = torch.tensor(0.0, device=var_pred.device)
        if cvar_pred is not None:
            cvar_pred = cvar_pred.squeeze()
            # CVaR should be less than VaR (more negative)
            cvar_loss = torch.mean(torch.relu(cvar_pred - var_pred))
        
        # ===================================================================
        # Total Loss
        # ===================================================================
        total_loss = nll_loss + self.lambda_cal * calibration_loss + \
                    self.lambda_cvar * cvar_loss
        
        return {
            'total': total_loss,
            'nll': nll_loss,
            'calibration': calibration_loss,
            'cvar': cvar_loss,
            'coverage': actual_coverage.detach().cpu().item()
        }


class BayesianVaRTrainer:
    """Bayesian VaR 모델 훈련"""
    
    def __init__(self, model: BayesianVaRNN, device: str = None):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.history = {
            'train_loss': [], 
            'val_loss': [],
            'train_calibration': [],
            'val_calibration': [],
            'train_coverage': [],
            'val_coverage': []
        }
    
    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer, 
                   criterion: BayesianVaRLoss, confidence: float = 0.95) -> Dict:
        """한 에포크 훈련"""
        self.model.train()
        epoch_metrics = {
            'loss': 0.0,
            'nll': 0.0,
            'calibration': 0.0,
            'coverage': 0.0
        }
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            optimizer.zero_grad()
            
            var_pred, aleatoric_std, cvar_pred = self.model(X_batch)
            loss_dict = criterion(var_pred, aleatoric_std, y_batch, cvar_pred, confidence)
            
            loss_dict['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_metrics['loss'] += loss_dict['total'].item()
            epoch_metrics['nll'] += loss_dict['nll'].item()
            epoch_metrics['calibration'] += loss_dict['calibration'].item()
            epoch_metrics['coverage'] += loss_dict['coverage']
        
        n_batches = len(train_loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= n_batches
        
        return epoch_metrics
    
    def validate(self, val_loader: DataLoader, criterion: BayesianVaRLoss,
                confidence: float = 0.95) -> Dict:
        """검증"""
        self.model.eval()
        val_metrics = {
            'loss': 0.0,
            'nll': 0.0,
            'calibration': 0.0,
            'coverage': 0.0
        }
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                var_pred, aleatoric_std, cvar_pred = self.model(X_batch)
                loss_dict = criterion(var_pred, aleatoric_std, y_batch, cvar_pred, confidence)
                
                val_metrics['loss'] += loss_dict['total'].item()
                val_metrics['nll'] += loss_dict['nll'].item()
                val_metrics['calibration'] += loss_dict['calibration'].item()
                val_metrics['coverage'] += loss_dict['coverage']
        
        n_batches = len(val_loader)
        for key in val_metrics:
            val_metrics[key] /= n_batches
        
        return val_metrics
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray,
            epochs: int = 100, batch_size: int = 256, 
            learning_rate: float = 0.001, patience: int = 15,
            confidence: float = 0.95) -> Dict:
        """모델 훈련"""
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        criterion = BayesianVaRLoss(lambda_cal=1.0, lambda_cvar=0.1)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print("="*80)
        print("TRAINING BAYESIAN VAR MODEL (FIXED NLL)")
        print("="*80)
        print(f"\nCorrect NLL formula: log(σ) + 0.5 * [(y - μ) / σ]²")
        print(f"Expected loss range: 0.0 ~ 2.0 (positive!)\n")
        
        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader, optimizer, criterion, confidence)
            val_metrics = self.validate(val_loader, criterion, confidence)
            
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_calibration'].append(train_metrics['calibration'])
            self.history['val_calibration'].append(val_metrics['calibration'])
            self.history['train_coverage'].append(train_metrics['coverage'])
            self.history['val_coverage'].append(val_metrics['coverage'])
            
            scheduler.step(val_metrics['loss'])
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {train_metrics['loss']:.6f} | "
                      f"Val Loss: {val_metrics['loss']:.6f} | "
                      f"Val Coverage: {val_metrics['coverage']:.2%} (Target: {confidence:.0%}) | "
                      f"Cal Error: {val_metrics['calibration']:.4f}")
            
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_bayesian_var_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        self.model.load_state_dict(torch.load('best_bayesian_var_model.pt'))
        print(f"\n✓ Training completed.")
        print(f"✓ Best val loss: {best_val_loss:.6f}")
        print(f"✓ Final val coverage: {self.history['val_coverage'][-1]:.2%} (Target: {confidence:.0%})")
        
        return self.history


def main():
    """Main execution"""
    import os
    
    print("Loading synthetic data...")
    data = np.load('../data/synthetic_data.npz')
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    
    print("\nData statistics:")
    print(f"  y_train: min={y_train.min():.4f}, max={y_train.max():.4f}, mean={y_train.mean():.4f}")
    print(f"  y_val:   min={y_val.min():.4f}, max={y_val.max():.4f}, mean={y_val.mean():.4f}")
    
    print("\nCreating Bayesian VaR model (FIXED NLL)...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BayesianVaRNN(input_dim=11, hidden_dim=128, dropout_rate=0.2)
    
    print("\nTraining...")
    trainer = BayesianVaRTrainer(model, device=device)
    history = trainer.fit(
        X_train, y_train, X_val, y_val,
        epochs=50,
        batch_size=256, 
        learning_rate=0.001,
        patience=10,
        confidence=0.95
    )
    
    return model, trainer, history


if __name__ == '__main__':
    model, trainer, history = main()
