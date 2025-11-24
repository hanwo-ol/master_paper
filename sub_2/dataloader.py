import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm

class MetaPortfolioDataset(Dataset):
    def __init__(self, k=4, tau=0.60, split='train', base_dir='./processed_data'):
        self.data_dir = os.path.join(base_dir, f'K{k}_tau{tau:.2f}')
        
        # 1. Load Metadata
        with open(os.path.join(self.data_dir, f'episodes_K{k}.json'), 'r') as f:
            all_episodes = json.load(f)
        
        self.episodes = [ep for ep in all_episodes if ep['split'] == split]
        print(f"[{split.upper()}] Loaded {len(self.episodes)} episodes")
        
        # 2. Load Panel Data
        print("Loading Panel Data...")
        panel_path = os.path.join(self.data_dir, f'sp500_panel_K{k}.csv')
        df = pd.read_csv(panel_path)
        
        # 날짜와 티커 매핑 생성
        # date 컬럼을 datetime으로 변환
        df['date'] = pd.to_datetime(df['date'])
        unique_dates = sorted(df['date'].unique())
        unique_tickers = sorted(df['ticker'].unique())
        
        self.date_to_idx = {d: i for i, d in enumerate(unique_dates)}
        self.ticker_to_idx = {t: i for i, t in enumerate(unique_tickers)}
        
        n_dates = len(unique_dates)
        n_assets = len(unique_tickers)
        
        # Feature 컬럼 추출
        exclude_cols = ['date', 'ticker', 'row_id', 'regime', 'returns']
        feature_cols = [c for c in df.columns if c not in exclude_cols and not c.endswith('_missing')]
        self.n_features = len(feature_cols)
        self.n_assets = n_assets
        
        print(f"Constructing Dense Tensor ({n_dates} days x {n_assets} assets x {self.n_features} features)...")
        
        # 3. Dense Tensor 생성 (Time, Asset, Feature)
        # 초기값은 0 (또는 NaN 처리 필요시 변경)
        self.X = torch.zeros(n_dates, n_assets, self.n_features, dtype=torch.float32)
        self.Y = torch.zeros(n_dates, n_assets, dtype=torch.float32)
        
        # DataFrame을 numpy로 변환하여 빠르게 채우기
        # date_idx, ticker_idx 컬럼 생성
        df['date_idx'] = df['date'].map(self.date_to_idx)
        df['ticker_idx'] = df['ticker'].map(self.ticker_to_idx)
        
        # 좌표와 값 추출
        rows = df['date_idx'].values
        cols = df['ticker_idx'].values
        feat_vals = df[feature_cols].values
        ret_vals = df['returns'].values
        
        # Tensor 할당
        self.X[rows, cols] = torch.tensor(feat_vals, dtype=torch.float32)
        self.Y[rows, cols] = torch.tensor(ret_vals, dtype=torch.float32)
        
        print("✓ Dense Tensor Construction Complete.")

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        ep = self.episodes[idx]
        
        # Episode의 시작/끝 날짜 문자열을 datetime으로 변환
        sup_start = pd.Timestamp(ep['support_start'])
        sup_end = pd.Timestamp(ep['support_end'])
        qry_start = pd.Timestamp(ep['query_start'])
        qry_end = pd.Timestamp(ep['query_end'])
        
        # Date Index 찾기
        # 주의: unique_dates 리스트에서 인덱스를 찾아야 함 (딕셔너리 활용)
        # support 기간의 모든 날짜 인덱스를 가져옴
        
        # 슬라이싱을 위해 시작 인덱스와 길이 계산
        s_idx = self.date_to_idx[sup_start]
        q_idx = self.date_to_idx[qry_start]
        
        sup_len = ep['n_support_days']
        qry_len = ep['n_query_days']
        
        # Slicing: (Time, Asset, Feature)
        sup_x = self.X[s_idx : s_idx + sup_len] 
        sup_y = self.Y[s_idx : s_idx + sup_len]
        
        qry_x = self.X[q_idx : q_idx + qry_len]
        qry_y = self.Y[q_idx : q_idx + qry_len]
        
        # Permute for Model Input: (Time, Asset, Feature) -> (Feature, Asset, Time)
        # Conv2d expects (Batch, Channel, Height, Width)
        # Here: Channel=Feature, Height=Asset, Width=Time
        
        sup_x = sup_x.permute(2, 1, 0) # (F, N, T_sup)
        qry_x = qry_x.permute(2, 1, 0) # (F, N, T_qry)
        
        return {
            'support_x': sup_x,
            'support_y': sup_y, # (T_sup, N)
            'query_x': qry_x,
            'query_y': qry_y,   # (T_qry, N)
            'regime': ep['support_regime'],
            'episode_id': ep['episode_id']
        }

if __name__ == "__main__":
    # Test
    ds = MetaPortfolioDataset(k=4, tau=0.60, split='train')
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    
    batch = next(iter(loader))
    print("\n[Batch Shape Check]")
    print("Support X:", batch['support_x'].shape) # (B, F, N, T)
    print("Support Y:", batch['support_y'].shape)
    print("Regimes:", batch['regime'])