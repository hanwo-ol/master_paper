# dataloader_ge.py

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
import os
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import StandardScaler

class HRPOrderer:
    """
    논문의 Step 1: Hierarchical Risk Parity (HRP) 기반 자산 재정렬 구현.
    Proposition 2에 따라 공분산 행렬의 구조적 안정성을 높이기 위해 사용.
    """
    def __init__(self, returns_df):
        """
        Args:
            returns_df (pd.DataFrame): (Time, Assets) 형태의 수익률 데이터 (Train 구간만 사용 권장)
        """
        self.returns_df = returns_df
        self.ordered_indices = self._compute_hrp_order()
        
    def _compute_hrp_order(self):
        # 1. 상관계수 행렬 계산
        corr = self.returns_df.corr().fillna(0)
        
        # 2. 거리 행렬 계산 (Distance metric)
        dist = np.sqrt((1 - corr) / 2)
        
        # 3. 계층적 군집화 (Linkage)
        # method='single' or 'ward' commonly used
        linkage = sch.linkage(dist, method='ward')
        
        # 4. Dendrogram에서 Leaf Ordering 추출 (Quasi-diagonalization)
        # 잎 노드의 순서가 군집화된 자산 순서임
        dendrogram = sch.dendrogram(linkage, no_plot=True)
        sort_ix = dendrogram['leaves']
        
        return sort_ix

    def apply_ordering(self, tensor_data):
        """
        Args:
            tensor_data: (..., Assets, ...) 형태의 텐서 또는 배열
        Returns:
            Reordered data
        """
        # 자산 축(Axis)이 어디인지에 따라 다르지만, 보통 N은 고정된 위치
        # 여기서는 numpy array indexing을 가정
        return tensor_data[..., self.ordered_indices, :]

class MetaPortfolioDataset(Dataset):
    """
    논문의 Multi-Task Meta-Learning을 위한 데이터셋.
    각 아이템은 하나의 'Episode' (Support Set + Query Set)를 반환.
    """
    def __init__(self, 
                 data_dir='./processed_data', 
                 k=3, 
                 tau=0.70, 
                 split='train', 
                 lookback_window=240,
                 n_assets=None, # None이면 자동 감지
                 target_col='returns'):
        
        super().__init__()
        
        # 경로 설정
        base_dir = os.path.join(data_dir, f'K{k}_tau{tau:.2f}')
        panel_path = os.path.join(base_dir, f'sp500_panel_K{k}.csv')
        episodes_path = os.path.join(base_dir, f'episodes_K{k}.json')
        meta_path = os.path.join(base_dir, f'metadata_K{k}.json')
        
        # 1. 메타데이터 및 에피소드 로드
        with open(meta_path, 'r') as f:
            self.metadata = json.load(f)
        
        with open(episodes_path, 'r') as f:
            all_episodes = json.load(f)
            
        # Split 필터링
        self.episodes = [ep for ep in all_episodes if ep['split'] == split]
        self.split = split
        self.lookback_window = lookback_window
        self.n_regimes = k
        
        # 2. 패널 데이터 로드
        print(f"Loading panel data from {panel_path}...")
        df = pd.read_csv(panel_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # Pivot Table로 변환: (Time, Assets, Features) 구조를 만들기 위함
        # Ticker 목록 추출
        self.tickers = sorted(df['ticker'].unique())
        if n_assets:
            self.tickers = self.tickers[:n_assets]
            df = df[df['ticker'].isin(self.tickers)]
        
        self.n_assets = len(self.tickers)
        
        # Feature 목록 (returns, date, ticker, row_id, regime 제외)
        exclude_cols = ['date', 'ticker', 'row_id', 'regime', 'returns']
        # returns는 target으로 별도 관리
        self.feature_cols = [c for c in df.columns if c not in exclude_cols]
        self.target_col = target_col
        
        print(f"Constructing 3D Tensor (Time x Assets x Features)...")
        # (Time, Assets, Features) 형태의 3D Array 생성
        # 날짜 정렬
        dates = sorted(df['date'].unique())
        self.date_to_idx = {d: i for i, d in enumerate(dates)}
        
        # 빠른 인덱싱을 위해 MultiIndex Pivot 후 Numpy 변환
        # shape: (T, N, F)
        df_sorted = df.sort_values(['date', 'ticker'])
        
        # Features
        X_raw = df_sorted.set_index(['date', 'ticker'])[self.feature_cols].unstack('ticker')
        # Reorder columns to match self.tickers just in case
        X_raw = X_raw.reindex(columns=self.tickers, level=1)
        self.X_tensor = X_raw.values.reshape(len(dates), self.n_assets, len(self.feature_cols))
        
        # Targets (Returns)
        y_raw = df_sorted.set_index(['date', 'ticker'])[self.target_col].unstack('ticker')
        y_raw = y_raw.reindex(columns=self.tickers)
        self.y_tensor = y_raw.values # (T, N)
        
        # Regimes (Time series of regimes) - for soft label generation
        # 날짜별 Regime은 모든 자산에 대해 동일하므로 첫 번째 자산 기준으로 추출
        regime_series = df_sorted.set_index(['date', 'ticker'])['regime'].unstack('ticker').iloc[:, 0]
        self.regime_tensor = regime_series.values # (T,)

        # 3. Feature Scaling (Train Split 기준)
        # Look-ahead bias 방지를 위해 Train 기간의 통계량만 사용해야 함
        train_end_date = pd.to_datetime(self.metadata['train_end_date'])
        train_idx = [i for i, d in enumerate(dates) if d <= train_end_date]
        
        # Reshape for scaling: (T_train * N, F)
        X_train_flat = self.X_tensor[train_idx].reshape(-1, len(self.feature_cols))
        
        self.scaler = StandardScaler()
        self.scaler.fit(X_train_flat)
        
        # 전체 데이터 스케일링 적용
        # (T, N, F) -> (T*N, F) -> transform -> (T, N, F)
        T, N, F = self.X_tensor.shape
        self.X_tensor = self.scaler.transform(self.X_tensor.reshape(-1, F)).reshape(T, N, F)
        
        # 4. HRP Ordering 계산 (Train 데이터만 사용)
        print("Computing HRP Asset Ordering...")
        train_returns = pd.DataFrame(self.y_tensor[train_idx], columns=self.tickers)
        self.hrp = HRPOrderer(train_returns)
        self.ordered_indices = self.hrp.ordered_indices
        
        # 텐서 재정렬 (Assets 차원: 1번 축)
        self.X_tensor = self.X_tensor[:, self.ordered_indices, :]
        self.y_tensor = self.y_tensor[:, self.ordered_indices]
        
        # 텐서를 PyTorch로 변환
        self.X_tensor = torch.FloatTensor(self.X_tensor)
        self.y_tensor = torch.FloatTensor(self.y_tensor)
        self.regime_tensor = torch.LongTensor(self.regime_tensor)
        
        print(f"Dataset initialized. Split: {split}, Episodes: {len(self.episodes)}")
        print(f"Tensor Shape: {self.X_tensor.shape}")

    def __len__(self):
        return len(self.episodes)

    def _get_window(self, end_row_id):
        """
        특정 시점(end_row_id)에서 Lookback Window만큼의 데이터를 가져옴.
        Input Shape for U-Net: (Features, Assets, Time)
        """
        # end_row_id는 해당 날짜의 index (0 ~ T-1)
        # 데이터는 end_row_id 포함 이전 lookback_window 개
        start_idx = end_row_id - self.lookback_window + 1
        
        if start_idx < 0:
            # Padding logic if needed, but preprocessing should ensure validity
            raise ValueError(f"Index {end_row_id} is too small for lookback {self.lookback_window}")
            
        # Slice: (Lookback, N, F)
        X_slice = self.X_tensor[start_idx : end_row_id + 1]
        
        # Permute to (F, N, Lookback) for Conv2d (Channels, Height, Width)
        # Height=Assets, Width=Time
        X_slice = X_slice.permute(2, 1, 0) 
        
        return X_slice

    def _make_soft_label(self, regime_idx, smoothing=0.1):
        """
        Hard label을 Soft probability vector로 변환 (Label Smoothing).
        추후 실제 HMM Posterior가 있다면 그것을 사용하도록 수정 가능.
        """
        prob = torch.full((self.n_regimes,), smoothing / (self.n_regimes - 1))
        prob[regime_idx] = 1.0 - smoothing
        return prob

    def __getitem__(self, idx):
        ep = self.episodes[idx]
        
        # Episode Metadata
        support_rows = ep['support_row_ids'] # List of indices
        query_rows = ep['query_row_ids']
        
        # 1. Support Set 구성
        # Batch of (X, y) pairs
        support_X = []
        support_y = []
        
        for r_id in support_rows:
            # X: (F, N, T_hist)
            x_window = self._get_window(r_id)
            # y: (N,) - Next day return (target is the return AT that day)
            # 주의: row_id가 가리키는 날짜의 return이 target임.
            # Preprocessing에서 row_id는 해당 날짜를 의미.
            y_val = self.y_tensor[r_id]
            
            support_X.append(x_window)
            support_y.append(y_val)
            
        support_X = torch.stack(support_X) # (Batch_S, F, N, T_hist)
        support_y = torch.stack(support_y) # (Batch_S, N)
        
        # 2. Query Set 구성
        query_X = []
        query_y = []
        
        for r_id in query_rows:
            x_window = self._get_window(r_id)
            y_val = self.y_tensor[r_id]
            
            query_X.append(x_window)
            query_y.append(y_val)
            
        query_X = torch.stack(query_X) # (Batch_Q, F, N, T_hist)
        query_y = torch.stack(query_y) # (Batch_Q, N)
        
        # 3. Regime Info (Soft Label)
        # 논문의 "Soft Task-Weighted"를 위해 Query 시점의 Regime 확률이 필요
        # 여기서는 Episode의 대표 Regime(query_regime)을 사용하되 Smoothing 적용
        regime_hard = ep['query_regime']
        regime_soft = self._make_soft_label(regime_hard)
        
        return {
            'episode_id': ep['episode_id'],
            'support_X': support_X,
            'support_y': support_y,
            'query_X': query_X,
            'query_y': query_y,
            'regime_hard': torch.tensor(regime_hard, dtype=torch.long),
            'regime_soft': regime_soft, # (K,)
            'support_regime': torch.tensor(ep['support_regime'], dtype=torch.long)
        }

def get_dataloader(data_dir, k, split, batch_size=4, num_workers=0):
    """
    Helper function to create DataLoader
    """
    dataset = MetaPortfolioDataset(data_dir=data_dir, k=k, split=split)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'), # Train만 Shuffle
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader

# --- Test Code ---
if __name__ == "__main__":
    # 간단한 테스트
    try:
        print("Testing DataLoader...")
        loader = get_dataloader(
            data_dir='./processed_data', 
            k=3, 
            split='train', 
            batch_size=2
        )
        
        batch = next(iter(loader))
        print("\nBatch Structure:")
        print(f"Support X: {batch['support_X'].shape} (Batch, Shots, Feat, Assets, Time)")
        print(f"Support y: {batch['support_y'].shape}")
        print(f"Query X:   {batch['query_X'].shape}")
        print(f"Regime Soft: {batch['regime_soft']}")
        print("✓ DataLoader test passed.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("Make sure you ran '005_preprocessing.py' first to generate data.")