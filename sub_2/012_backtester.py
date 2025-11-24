import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import copy
from tqdm import tqdm
from scipy.optimize import minimize

# 모듈 임포트
from model_architecture import AssetUNet

class Backtester:
    def __init__(self, k=4, tau=0.60, meta_model_path=None, pooled_model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.k = k
        self.tau = tau
        
        # 1. 전체 Panel Data 로드 (순차적 접근을 위해)
        data_dir = f'./processed_data/K{k}_tau{tau:.2f}'
        panel_path = os.path.join(data_dir, f'sp500_panel_K{k}.csv')
        self.df = pd.read_csv(panel_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Pivot Data (Time x Asset x Feature)
        # 메모리 효율을 위해 필요한 컬럼만 추출 및 정렬
        self.unique_dates = sorted(self.df['date'].unique())
        self.unique_tickers = sorted(self.df['ticker'].unique())
        self.n_assets = len(self.unique_tickers)
        
        # Feature Columns
        exclude_cols = ['date', 'ticker', 'row_id', 'regime', 'returns']
        self.feature_cols = [c for c in self.df.columns if c not in exclude_cols and not c.endswith('_missing')]
        self.n_features = len(self.feature_cols)
        
        # Tensor 변환 (전체 데이터를 메모리에 올림)
        print("Constructing Full Tensor for Backtest...")
        n_dates = len(self.unique_dates)
        
        self.X = torch.zeros(n_dates, self.n_assets, self.n_features, dtype=torch.float32)
        self.Y = torch.zeros(n_dates, self.n_assets, dtype=torch.float32)
        
        # Mapping
        date_map = {d: i for i, d in enumerate(self.unique_dates)}
        ticker_map = {t: i for i, t in enumerate(self.unique_tickers)}
        
        self.df['date_idx'] = self.df['date'].map(date_map)
        self.df['ticker_idx'] = self.df['ticker'].map(ticker_map)
        
        rows = self.df['date_idx'].values
        cols = self.df['ticker_idx'].values
        
        self.X[rows, cols] = torch.tensor(self.df[self.feature_cols].values, dtype=torch.float32)
        self.Y[rows, cols] = torch.tensor(self.df['returns'].values, dtype=torch.float32)
        
        # Test Period 설정 (Metadata 참조)
        with open(os.path.join(data_dir, f'metadata_K{k}.json')) as f:
            meta = json.load(f)
            self.val_end_date = pd.Timestamp(meta['val_end_date'])
            
        # Test Start Index
        # Val End Date 다음 날부터 시작
        self.test_start_idx = date_map[self.val_end_date] + 1
        print(f"Backtest Start Date: {self.unique_dates[self.test_start_idx]}")
        print(f"Total Test Days: {n_dates - self.test_start_idx}")
        
        # 모델 로드
        self.meta_model = AssetUNet(self.n_features, self.n_assets).to(self.device)
        if meta_model_path:
            self.meta_model.load_state_dict(torch.load(meta_model_path, map_location=self.device))
            print("✓ Meta-Model Loaded")
            
        self.pooled_model = AssetUNet(self.n_features, self.n_assets).to(self.device)
        if pooled_model_path:
            self.pooled_model.load_state_dict(torch.load(pooled_model_path, map_location=self.device))
            print("✓ Pooled-Model Loaded")

    def get_window_data(self, current_idx, window_size=60):
        """
        현재 시점(current_idx) 기준 과거 window_size 만큼의 데이터를 가져옴
        """
        if current_idx < window_size:
            return None, None
            
        # Slice: [current_idx - window_size : current_idx]
        # X shape: (Window, Asset, Feature) -> (Feature, Asset, Window) for Model
        x_window = self.X[current_idx-window_size : current_idx]
        y_window = self.Y[current_idx-window_size : current_idx]
        
        x_tensor = x_window.permute(2, 1, 0).unsqueeze(0).to(self.device) # (1, F, N, T)
        y_tensor = y_window.to(self.device) # (T, N)
        
        return x_tensor, y_tensor

    def adapt_meta_model(self, support_x, support_y):
        """
        Test-Time Adaptation (MAML Inference)
        """
        fast_model = copy.deepcopy(self.meta_model)
        fast_model.train()
        # Inner LR (Training과 동일하게 설정)
        optimizer = optim.SGD(fast_model.parameters(), lr=0.001)
        
        # Support Y의 마지막 시점만 Target으로 쓰는게 아니라, 
        # Support Set 전체 시퀀스에 대해 적응하거나, 마지막 시점만 쓰거나 선택 가능.
        # MAML 학습 시에는 Last Step Target을 썼으므로 여기서도 맞춤.
        target = support_y[-1, :].unsqueeze(0) # (1, N)
        
        for _ in range(5): # 5 Gradient Steps
            pred = fast_model(support_x) # (1, N)
            loss = nn.MSELoss()(pred, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        fast_model.eval()
        return fast_model

    def optimize_portfolio(self, mu, cov, prev_w, kappa=0.0005):
        """
        Constrained Mean-Variance Optimization
        """
        n = len(mu)
        # Objective: Maximize Utility (Ret - Risk - Cost)
        # Minimize: 0.5 * wCw - wMu + kappa * |dw|
        
        # Covariance Regularization (Shrinkage)
        cov = cov + np.eye(n) * 1e-5
        
        def objective(w):
            risk = 0.5 * np.dot(w.T, np.dot(cov, w))
            ret = np.dot(w.T, mu)
            cost = kappa * np.sum(np.abs(w - prev_w))
            return risk - ret + cost
        
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((0, 1) for _ in range(n))
        w0 = prev_w if prev_w is not None else np.ones(n)/n
        
        try:
            res = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints, tol=1e-4)
            return res.x
        except:
            return w0

    def run_backtest(self):
        print("="*80)
        print("Running Strict Rolling Window Backtest...")
        print("="*80)
        
        results = {
            'Meta': {'ret': [], 'to': []},
            'Pooled': {'ret': [], 'to': []},
            'EW': {'ret': [], 'to': []}
        }
        
        prev_w_meta = np.ones(self.n_assets) / self.n_assets
        prev_w_pooled = np.ones(self.n_assets) / self.n_assets
        
        # Test Loop
        # test_start_idx 부터 끝까지 하루씩 전진
        for t in tqdm(range(self.test_start_idx, len(self.unique_dates) - 1)):
            # 1. Data Preparation (Lookback 60 days)
            # t 시점에서의 정보는 t-1까지의 데이터임.
            # 하지만 우리 데이터셋 구조상 X[t]는 t 시점의 Feature, Y[t]는 t+1 시점의 Return임.
            # 따라서 t 시점에서 예측을 하려면 X[t-59 : t+1] (60일치)를 가져와야 함.
            
            # Support Set: t-60 ~ t-1 (과거 60일)
            sup_x, sup_y = self.get_window_data(t, window_size=60)
            
            # Query Input: t-59 ~ t (현재 시점 포함 60일) -> 다음날(t+1) 예측용
            # 모델은 항상 고정된 윈도우 크기(60)를 입력으로 받음
            qry_x, _ = self.get_window_data(t + 1, window_size=60)
            
            if sup_x is None or qry_x is None: continue
            
            # 2. Prediction
            # (A) Meta-Model
            fast_model = self.adapt_meta_model(sup_x, sup_y)
            with torch.no_grad():
                pred_meta = fast_model(qry_x).cpu().numpy().flatten()
                
            # (B) Pooled Model
            self.pooled_model.eval()
            with torch.no_grad():
                pred_pooled = self.pooled_model(qry_x).cpu().numpy().flatten()
                
            # 3. Risk Model (Covariance from Support Set Returns)
            sup_ret = sup_y.cpu().numpy() # (60, N)
            cov = np.cov(sup_ret, rowvar=False)
            
            # 4. Optimization
            w_meta = self.optimize_portfolio(pred_meta, cov, prev_w_meta)
            w_pooled = self.optimize_portfolio(pred_pooled, cov, prev_w_pooled)
            w_ew = np.ones(self.n_assets) / self.n_assets
            
            # 5. Evaluation (Realized Return at t+1)
            # Y[t]가 t+1의 수익률임 (데이터셋 정의상)
            real_ret = self.Y[t].numpy()
            
            r_meta = np.dot(w_meta, real_ret)
            r_pooled = np.dot(w_pooled, real_ret)
            r_ew = np.dot(w_ew, real_ret)
            
            # Turnover
            to_meta = np.sum(np.abs(w_meta - prev_w_meta))
            to_pooled = np.sum(np.abs(w_pooled - prev_w_pooled))
            
            # Update Weights
            prev_w_meta = w_meta
            prev_w_pooled = w_pooled
            
            # Log
            results['Meta']['ret'].append(r_meta)
            results['Meta']['to'].append(to_meta)
            results['Pooled']['ret'].append(r_pooled)
            results['Pooled']['to'].append(to_pooled)
            results['EW']['ret'].append(r_ew)
            results['EW']['to'].append(0)
            
        # Save Results
        df_res = pd.DataFrame()
        df_res['Date'] = self.unique_dates[self.test_start_idx : self.test_start_idx + len(results['Meta']['ret'])]
        
        for strat in results:
            df_res[f'{strat}_Return'] = results[strat]['ret']
            df_res[f'{strat}_Turnover'] = results[strat]['to']
            
        os.makedirs('./results', exist_ok=True)
        df_res.to_csv('./results/backtest_results_strict.csv', index=False)
        print("✓ Strict Backtest Complete.")
        
        return df_res

if __name__ == "__main__":
    import json
    # 경로 설정
    meta_path = './checkpoints/K4_tau0.60/best_meta_model.pth'
    pooled_path = './checkpoints/pooled_K4_tau0.60/best_pooled_model.pth'
    
    backtester = Backtester(k=4, tau=0.60, meta_model_path=meta_path, pooled_model_path=pooled_path)
    backtester.run_backtest()