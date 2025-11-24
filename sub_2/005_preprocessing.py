import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

class BatchPreprocessor:
    def __init__(self, panel_path='./sp500_data/sp500_panel.csv', 
                 output_base_dir='./processed_data'):
        self.panel_path = panel_path
        self.output_base_dir = output_base_dir
        self.panel_data = None
        self.daily_summary = None
        
        self.train_end_date = None
        self.val_end_date = None
        
        os.makedirs(output_base_dir, exist_ok=True)
    
    def determine_split_dates(self):
        print("="*80)
        print("Split 날짜 결정")
        print("="*80)
        
        df = self.panel_data
        unique_dates = sorted(df.index.unique())
        
        support_len = 60
        query_len = 60
        min_history = 252
        
        start_idx = min_history + support_len
        end_idx = len(unique_dates) - query_len
        available_starts = unique_dates[start_idx:end_idx]
        
        n_total = len(available_starts)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.15)
        
        self.train_end_date = available_starts[n_train - 1]
        self.val_end_date = available_starts[n_train + n_val - 1]
        
        print(f"Split 결정: Train End={self.train_end_date}, Val End={self.val_end_date}")
        return self.train_end_date, self.val_end_date
    
    def load_and_prepare_data(self):
        print("\n" + "="*80)
        print("데이터 로드 및 전처리")
        print("="*80)
        
        df = pd.read_csv(self.panel_path, index_col=0, parse_dates=True)
        
        if 'asset_turnover' in df.columns:
            df = df.drop(columns=['asset_turnover'])
        
        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: 'date'})
        df['date'] = pd.to_datetime(df['date'])
        df['row_id'] = np.arange(len(df))
        df = df.set_index('date')
        
        self.panel_data = df
        self.determine_split_dates()
        return df
    
    def impute_with_train_only(self):
        print("\n" + "="*80)
        print(f"Train-Only Imputation & Scaling")
        print("="*80)
        
        df = self.panel_data
        
        # [1] Inf 제거 (무한대를 NaN으로 변환)
        print("Replacing Inf with NaN...")
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Train 구간 정의
        train_mask = df.index <= self.train_end_date
        train_df = df[train_mask]
        
        # [2] Missing Indicator 추가
        fundamental_cols = ['pe_ratio', 'pb_ratio', 'roe', 'roa', 'debt_to_equity',
                           'dividend_yield', 'payout_ratio', 'free_cashflow_yield']
        fundamental_cols = [col for col in fundamental_cols if col in df.columns]
        
        for col in fundamental_cols:
            df[f'{col}_missing'] = df[col].isnull().astype(int)
        
        # [3] Imputation
        print("Imputing missing values...")
        
        # Fundamental: Train Median
        for col in fundamental_cols:
            train_median = train_df[col].median()
            df[col] = df.groupby('ticker')[col].transform(lambda x: x.fillna(method='ffill'))
            df[col] = df[col].fillna(train_median)
        
        # Macro: Forward Fill
        macro_cols = ['vix', 'treasury_10y', 'treasury_2y', 'yield_spread',
                     'usd_index', 'credit_spread', 'cpi_yoy']
        for col in macro_cols:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        # Technical: Ticker-wise Forward Fill
        tech_cols = ['rsi', 'ma_20', 'ma_50', 'ma_200', 'macd', 'macd_signal',
                    'bollinger_upper', 'bollinger_lower', 'atr', 'stoch_k', 'stoch_d',
                    'price_roc', 'realized_vol_20', 'realized_vol_60', 'volume_roc',
                    'mfi', 'obv', 'williams_r']
        
        # OHLCV도 포함
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        
        all_tech_cols = tech_cols + ohlcv_cols
        for col in all_tech_cols:
            if col in df.columns:
                df[col] = df.groupby('ticker')[col].transform(
                    lambda x: x.fillna(method='ffill').fillna(method='bfill')
                )
                # 여전히 남은 NaN은 0으로 채움 (시작점 등)
                df[col] = df[col].fillna(0)

        # Returns NaN 처리 (0으로)
        df['returns'] = df['returns'].fillna(0)
        
        # [4] Feature Scaling (Standardization)
        print("Applying StandardScaler (fit on Train)...")
        
        # 스케일링 대상 컬럼 (returns, ticker, row_id, date, missing_indicator 제외)
        exclude_cols = ['ticker', 'returns', 'row_id', 'date']
        feature_cols = [c for c in df.columns if c not in exclude_cols and not c.endswith('_missing')]
        
        # Train 데이터로 Scaler 학습
        scaler = StandardScaler()
        # Train 데이터 중 NaN이 없는 부분만 사용하여 fit
        train_features = df.loc[train_mask, feature_cols]
        scaler.fit(train_features)
        
        # 전체 데이터 변환
        df[feature_cols] = scaler.transform(df[feature_cols])
        
        # [5] Returns Clipping (Extreme Value 처리)
        # 수익률이 -90% ~ +100% 범위를 벗어나면 클리핑
        df['returns'] = df['returns'].clip(lower=-0.9, upper=1.0)
        
        print("✓ Preprocessing & Scaling Complete.")
        self.panel_data = df
        return df
    
    def compute_market_summary(self):
        print("\n" + "="*80)
        print("Market Summary 계산")
        print("="*80)
        
        df = self.panel_data
        
        # 주의: df는 이미 스케일링되어 있으므로, Market Summary용으로는 
        # 스케일링 전 원본 값이나 Returns를 써야 함.
        # 여기서는 Returns와 VIX(스케일링됨) 등을 그대로 사용하되, 
        # Clustering에 쓰이는 Feature이므로 일관성만 있으면 됨.
        
        daily_summary = df.groupby(df.index).agg({
            'returns': ['mean', 'std'],
            'volume': 'sum', # Scaled volume sum (not physical volume, but acceptable for pattern)
            'vix': 'first',
            'treasury_10y': 'first',
            'yield_spread': 'first',
            'close': 'count'
        }).copy()
        
        daily_summary.columns = ['_'.join(col).strip() for col in daily_summary.columns.values]
        daily_summary.rename(columns={
            'returns_mean': 'market_return',
            'returns_std': 'market_volatility',
            'volume_sum': 'total_volume',
            'vix_first': 'vix',
            'treasury_10y_first': 'treasury_10y',
            'yield_spread_first': 'yield_spread',
            'close_count': 'n_assets'
        }, inplace=True)
        
        # Rolling Features
        for window in [5, 20]:
            daily_summary[f'ma_return_{window}d'] = daily_summary['market_return'].rolling(window).mean()
            daily_summary[f'ma_vol_{window}d'] = daily_summary['market_volatility'].rolling(window).mean()
        
        daily_summary = daily_summary.dropna()
        self.daily_summary = daily_summary
        return daily_summary
    
    def cluster_regimes_train_only(self, k):
        print(f"\nTrain-Only Regime Clustering (K={k})")
        daily = self.daily_summary
        
        features = [
            'market_return', 'market_volatility', 'vix',
            'treasury_10y', 'yield_spread',
            'ma_return_5d', 'ma_vol_5d',
            'ma_return_20d', 'ma_vol_20d'
        ]
        
        X = daily[features].values
        
        train_mask = daily.index <= self.train_end_date
        val_mask = (daily.index > self.train_end_date) & (daily.index <= self.val_end_date)
        test_mask = daily.index > self.val_end_date
        
        X_train = X[train_mask]
        X_val = X[val_mask]
        X_test = X[test_mask]
        
        # Market Summary도 다시 스케일링 (K-Means용)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        kmeans.fit(X_train_scaled)
        
        regime_labels = np.full(len(daily), -1)
        regime_labels[train_mask] = kmeans.predict(X_train_scaled)
        
        if val_mask.sum() > 0:
            regime_labels[val_mask] = kmeans.predict(scaler.transform(X_val))
        if test_mask.sum() > 0:
            regime_labels[test_mask] = kmeans.predict(scaler.transform(X_test))
            
        daily_with_regime = daily.copy()
        daily_with_regime['regime'] = regime_labels
        
        return daily_with_regime, kmeans, scaler
    
    def create_episodes(self, k, daily_summary, purity_threshold=0.70, support_len=60, query_len=60):
        # Panel에 regime 병합
        if 'regime' in self.panel_data.columns:
            df = self.panel_data.drop(columns=['regime'])
        else:
            df = self.panel_data.copy()
        
        df = df.join(daily_summary[['regime']], how='left')
        unique_dates = sorted(df.index.unique())
        
        min_history = 252
        start_idx = min_history + support_len
        end_idx = len(unique_dates) - query_len
        available_starts = unique_dates[start_idx:end_idx]
        
        train_dates = [d for d in available_starts if d <= self.train_end_date]
        val_dates = [d for d in available_starts if self.train_end_date < d <= self.val_end_date]
        test_dates = [d for d in available_starts if d > self.val_end_date]
        
        print(f"Episode 생성: Train={len(train_dates)}, Val={len(val_dates)}, Test={len(test_dates)}")
        
        def create_episode_list(date_list, split_name):
            episodes = []
            skipped = 0
            
            for start_date in tqdm(date_list, desc=f"{split_name}"):
                start_idx_local = unique_dates.index(start_date) - support_len
                support_dates = unique_dates[start_idx_local:start_idx_local + support_len]
                query_start_idx = start_idx_local + support_len
                query_dates = unique_dates[query_start_idx:query_start_idx + query_len]
                
                support_mask = df.index.isin(support_dates)
                query_mask = df.index.isin(query_dates)
                
                support_regimes = df.loc[support_mask, 'regime'].dropna()
                query_regimes = df.loc[query_mask, 'regime'].dropna()
                
                if len(support_regimes) == 0 or len(query_regimes) == 0:
                    skipped += 1
                    continue
                
                sup_mode = support_regimes.mode()[0]
                qry_mode = query_regimes.mode()[0]
                
                sup_purity = (support_regimes == sup_mode).mean()
                qry_purity = (query_regimes == qry_mode).mean()
                
                if sup_mode != qry_mode or sup_purity < purity_threshold or qry_purity < purity_threshold:
                    skipped += 1
                    continue
                
                support_row_ids = df.loc[support_mask, 'row_id'].tolist()
                query_row_ids = df.loc[query_mask, 'row_id'].tolist()
                
                episode = {
                    'episode_id': len(episodes),
                    'split': split_name,
                    'support_start': str(support_dates[0]),
                    'support_end': str(support_dates[-1]),
                    'query_start': str(query_dates[0]),
                    'query_end': str(query_dates[-1]),
                    'support_regime': int(sup_mode),
                    'support_row_ids': support_row_ids,
                    'query_row_ids': query_row_ids,
                    'n_support_days': len(support_dates),
                    'n_query_days': len(query_dates)
                }
                episodes.append(episode)
            return episodes, skipped
        
        train_episodes, train_skipped = create_episode_list(train_dates, 'train')
        val_episodes, val_skipped = create_episode_list(val_dates, 'val')
        test_episodes, test_skipped = create_episode_list(test_dates, 'test')
        
        all_episodes = train_episodes + val_episodes + test_episodes
        
        return {
            'episodes': all_episodes,
            'panel_with_regime': df,
            'stats': {'n_train': len(train_episodes), 'n_val': len(val_episodes), 'n_test': len(test_episodes)}
        }
    
    def save_results(self, k, daily_summary, episodes_data, kmeans, scaler, purity_threshold):
        output_dir = os.path.join(self.output_base_dir, f'K{k}_tau{purity_threshold:.2f}')
        os.makedirs(output_dir, exist_ok=True)
        
        episodes_data['panel_with_regime'].to_csv(os.path.join(output_dir, f'sp500_panel_K{k}.csv'))
        daily_summary.to_csv(os.path.join(output_dir, f'market_regimes_K{k}.csv'))
        
        with open(os.path.join(output_dir, f'episodes_K{k}.json'), 'w') as f:
            json.dump(episodes_data['episodes'], f, indent=2)
            
        import pickle
        with open(os.path.join(output_dir, f'kmeans_model_K{k}.pkl'), 'wb') as f:
            pickle.dump({'kmeans': kmeans, 'scaler': scaler}, f)
            
        return output_dir

    def batch_process(self, k_values=[4], purity_values=[0.60]):
        self.load_and_prepare_data()
        self.impute_with_train_only()
        self.compute_market_summary()
        
        for k in k_values:
            for tau in purity_values:
                print(f"\nProcessing K={k}, tau={tau}")
                daily, kmeans, scaler = self.cluster_regimes_train_only(k)
                ep_data = self.create_episodes(k, daily, tau)
                self.save_results(k, daily, ep_data, kmeans, scaler, tau)

if __name__ == "__main__":
    processor = BatchPreprocessor()
    # 디폴트 설정인 K=4, tau=0.60만 다시 실행
    processor.batch_process(k_values=[4], purity_values=[0.60])