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
    """
    개선된 배치 전처리:
    1. Train-only regime clustering (미래 정보 누출 방지)
    2. 통일된 split 날짜 사용
    """
    
    def __init__(self, panel_path='./sp500_data/sp500_panel.csv', 
                 output_base_dir='./processed_data'):
        self.panel_path = panel_path
        self.output_base_dir = output_base_dir
        self.panel_data = None
        self.daily_summary = None
        
        # Split 날짜 (시간 순서로 70/15/15)
        self.train_end_date = None
        self.val_end_date = None
        
        os.makedirs(output_base_dir, exist_ok=True)
    
    def determine_split_dates(self):
        """
        Episode 생성 가능한 날짜 기준으로 Train/Val/Test split 날짜 결정
        이후 모든 단계(imputation, regime clustering, episode)에서 동일 날짜 사용
        """
        print("="*80)
        print("Split 날짜 결정")
        print("="*80)
        
        df = self.panel_data
        unique_dates = sorted(df.index.unique())
        
        support_len = 60
        query_len = 60
        min_history = 252
        
        # Episode 생성 가능 날짜
        start_idx = min_history + support_len
        end_idx = len(unique_dates) - query_len
        available_starts = unique_dates[start_idx:end_idx]
        
        # 70/15/15 split
        n_total = len(available_starts)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.15)
        
        self.train_end_date = available_starts[n_train - 1]  # Train 마지막 날
        self.val_end_date = available_starts[n_train + n_val - 1]  # Val 마지막 날
        
        print(f"\n전체 episode 시작 가능 날짜: {len(available_starts)}")
        print(f"  첫 날짜: {available_starts[0]}")
        print(f"  마지막 날짜: {available_starts[-1]}")
        
        print(f"\nSplit 결정:")
        print(f"  Train: {available_starts[0]} ~ {self.train_end_date} ({n_train} slots)")
        print(f"  Val:   {available_starts[n_train]} ~ {self.val_end_date} ({n_val} slots)")
        print(f"  Test:  {available_starts[n_train + n_val]} ~ {available_starts[-1]} ({n_total - n_train - n_val} slots)")
        
        print(f"\n✓ 이 날짜를 모든 전처리 단계(imputation, regime, episode)에서 일관되게 사용")
        
        return self.train_end_date, self.val_end_date
    
    def load_and_prepare_data(self):
        """데이터 로드 및 기본 전처리"""
        print("\n" + "="*80)
        print("데이터 로드 및 전처리")
        print("="*80)
        
        # 로드
        df = pd.read_csv(self.panel_path, index_col=0, parse_dates=True)
        print(f"\n원본 shape: {df.shape}")
        
        # 1. asset_turnover 제거
        if 'asset_turnover' in df.columns:
            df = df.drop(columns=['asset_turnover'])
            print("✓ asset_turnover 제거")
        
        # Row ID 추가
        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: 'date'})
        df['date'] = pd.to_datetime(df['date'])
        df['row_id'] = np.arange(len(df))
        df = df.set_index('date')
        
        self.panel_data = df
        
        # Split 날짜 결정 (imputation 전에)
        self.determine_split_dates()
        
        return df
    
    def impute_with_train_only(self):
        """Train-only imputation (미래 정보 누출 방지)"""
        print("\n" + "="*80)
        print(f"Train-Only Imputation")
        print("="*80)
        
        df = self.panel_data
        
        # Train 구간
        train_mask = df.index <= self.train_end_date
        train_df = df[train_mask]
        
        print(f"\nTrain 구간: {train_df.index.min()} ~ {train_df.index.max()}")
        print(f"Train 데이터: {len(train_df):,} / {len(df):,} ({len(train_df)/len(df)*100:.1f}%)")
        
        # 2. Missing indicator 추가
        fundamental_cols = ['pe_ratio', 'pb_ratio', 'roe', 'roa', 'debt_to_equity',
                           'dividend_yield', 'payout_ratio', 'free_cashflow_yield']
        fundamental_cols = [col for col in fundamental_cols if col in df.columns]
        
        print(f"\n[1] Missing indicators 추가 ({len(fundamental_cols)}개)")
        for col in fundamental_cols:
            df[f'{col}_missing'] = df[col].isnull().astype(int)
        
        # 3. Train-only imputation
        print(f"\n[2] Train-only imputation")
        
        # Fundamental (train median)
        for col in fundamental_cols:
            train_median = train_df[col].median()
            
            # Ticker별 forward fill
            df[col] = df.groupby('ticker')[col].transform(lambda x: x.fillna(method='ffill'))
            
            # Train median으로 나머지 채우기
            df[col] = df[col].fillna(train_median)
            
            print(f"  {col}: median={train_median:.4f}")
        
        # Macro (forward fill만, 날짜 순서 유지)
        macro_cols = ['vix', 'treasury_10y', 'treasury_2y', 'yield_spread',
                     'usd_index', 'credit_spread', 'cpi_yoy']
        for col in macro_cols:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        # Technical (ticker별 forward fill)
        tech_cols = ['rsi', 'ma_20', 'ma_50', 'ma_200', 'macd', 'macd_signal',
                    'bollinger_upper', 'bollinger_lower', 'atr', 'stoch_k', 'stoch_d',
                    'price_roc', 'realized_vol_20', 'realized_vol_60', 'volume_roc',
                    'mfi', 'obv', 'williams_r']
        for col in tech_cols:
            if col in df.columns:
                df[col] = df.groupby('ticker')[col].transform(
                    lambda x: x.fillna(method='ffill').fillna(method='bfill')
                )
        
        final_missing = df.isnull().sum().sum()
        feature_count = len([col for col in df.columns 
                           if col not in ['ticker', 'returns', 'row_id']])
        
        print(f"\n✓ 최종 missing: {final_missing:,}")
        print(f"✓ 최종 feature 수: {feature_count}")
        
        self.panel_data = df
        return df
    
    def compute_market_summary(self):
        """Market summary 계산"""
        print("\n" + "="*80)
        print("Market Summary 계산")
        print("="*80)
        
        df = self.panel_data
        
        daily_summary = df.groupby(df.index).agg({
            'returns': ['mean', 'std'],
            'volume': 'sum',
            'vix': 'first',
            'treasury_10y': 'first',
            'treasury_2y': 'first',
            'yield_spread': 'first',
            'usd_index': 'first',
            'close': 'count'
        }).copy()
        
        daily_summary.columns = ['_'.join(col).strip() for col in daily_summary.columns.values]
        daily_summary.rename(columns={
            'returns_mean': 'market_return',
            'returns_std': 'market_volatility',
            'volume_sum': 'total_volume',
            'vix_first': 'vix',
            'treasury_10y_first': 'treasury_10y',
            'treasury_2y_first': 'treasury_2y',
            'yield_spread_first': 'yield_spread',
            'usd_index_first': 'usd_index',
            'close_count': 'n_assets'
        }, inplace=True)
        
        daily_summary['volume_change'] = daily_summary['total_volume'].pct_change()
        daily_summary['vix_change'] = daily_summary['vix'].pct_change()
        
        for window in [5, 20]:
            daily_summary[f'ma_return_{window}d'] = daily_summary['market_return'].rolling(window).mean()
            daily_summary[f'ma_vol_{window}d'] = daily_summary['market_volatility'].rolling(window).mean()
        
        daily_summary = daily_summary.dropna()
        
        print(f"✓ Daily summary: {daily_summary.shape}")
        
        self.daily_summary = daily_summary
        return daily_summary
    
    def cluster_regimes_train_only(self, k):
        """
        Train-only regime clustering (미래 정보 누출 방지)
        
        1. Train 구간 데이터로 K-means 학습
        2. Val/Test 구간은 학습된 모델로 predict
        """
        print("\n" + "="*80)
        print(f"Train-Only Regime Clustering (K={k})")
        print("="*80)
        
        daily = self.daily_summary
        
        # Clustering features
        features = [
            'market_return', 'market_volatility', 'vix',
            'treasury_10y', 'yield_spread',
            'ma_return_5d', 'ma_vol_5d',
            'ma_return_20d', 'ma_vol_20d'
        ]
        
        X = daily[features].values
        
        # Train/Val/Test split
        train_mask = daily.index <= self.train_end_date
        val_mask = (daily.index > self.train_end_date) & (daily.index <= self.val_end_date)
        test_mask = daily.index > self.val_end_date
        
        X_train = X[train_mask]
        X_val = X[val_mask]
        X_test = X[test_mask]
        
        print(f"\nData split:")
        print(f"  Train: {train_mask.sum()} days ({train_mask.sum()/len(daily)*100:.1f}%)")
        print(f"  Val:   {val_mask.sum()} days ({val_mask.sum()/len(daily)*100:.1f}%)")
        print(f"  Test:  {test_mask.sum()} days ({test_mask.sum()/len(daily)*100:.1f}%)")
        
        # Scaler fitting (Train만)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # K-means fitting (Train만)
        print(f"\n✓ K-means 학습 (Train 구간만, K={k})")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        kmeans.fit(X_train_scaled)
        
        # Train regime 할당
        regime_labels = np.full(len(daily), -1)  # Initialize
        regime_labels[train_mask] = kmeans.predict(X_train_scaled)
        
        # Val/Test regime 예측 (학습된 모델 사용)
        if val_mask.sum() > 0:
            X_val_scaled = scaler.transform(X_val)
            regime_labels[val_mask] = kmeans.predict(X_val_scaled)
            print(f"✓ Val regime 예측 완료")
        
        if test_mask.sum() > 0:
            X_test_scaled = scaler.transform(X_test)
            regime_labels[test_mask] = kmeans.predict(X_test_scaled)
            print(f"✓ Test regime 예측 완료")
        
        # Daily summary에 추가
        daily_with_regime = daily.copy()
        daily_with_regime['regime'] = regime_labels
        
        # Regime 분포
        print(f"\n[Regime 분포]")
        for split_name, mask in [('Train', train_mask), ('Val', val_mask), ('Test', test_mask)]:
            split_regimes = regime_labels[mask]
            if len(split_regimes) > 0:
                print(f"\n{split_name}:")
                for regime_id in range(k):
                    count = (split_regimes == regime_id).sum()
                    pct = count / len(split_regimes) * 100
                    print(f"  Regime {regime_id}: {count} days ({pct:.1f}%)")
        
        # Regime 특성 (Train 기준)
        print(f"\n[Regime 특성 (Train 기준)]")
        train_daily = daily_with_regime[train_mask]
        regime_chars = train_daily.groupby('regime')[
            ['market_return', 'market_volatility', 'vix', 'treasury_10y', 'yield_spread']
        ].mean()
        print(regime_chars.round(4))
        
        return daily_with_regime, kmeans, scaler
    
    def create_episodes(self, k, daily_summary, purity_threshold=0.70,
                       support_len=60, query_len=60):
        """Episode 생성 (통일된 split 날짜 사용)"""
        
        # Panel에 regime 병합
        if 'regime' in self.panel_data.columns:
            df = self.panel_data.drop(columns=['regime'])
        else:
            df = self.panel_data.copy()
        
        df = df.join(daily_summary[['regime']], how='left')
        
        unique_dates = sorted(df.index.unique())
        
        # Episode 생성 가능 날짜
        min_history = 252
        start_idx = min_history + support_len
        end_idx = len(unique_dates) - query_len
        available_starts = unique_dates[start_idx:end_idx]
        
        # Train/Val/Test split (이미 결정된 날짜 사용)
        train_dates = [d for d in available_starts if d <= self.train_end_date]
        val_dates = [d for d in available_starts if self.train_end_date < d <= self.val_end_date]
        test_dates = [d for d in available_starts if d > self.val_end_date]
        
        print(f"\n" + "="*80)
        print(f"Episode 생성 (K={k}, purity={purity_threshold})")
        print("="*80)
        print(f"\nSplit (determined dates):")
        print(f"  Train: {len(train_dates)} slots")
        print(f"  Val:   {len(val_dates)} slots")
        print(f"  Test:  {len(test_dates)} slots")
        
        def create_episode_list(date_list, split_name):
            episodes = []
            skipped = 0
            
            for start_date in tqdm(date_list, desc=f"{split_name} (K={k})"):
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
                
                # Regime-pure 조건
                if sup_mode != qry_mode:
                    skipped += 1
                    continue
                if sup_purity < purity_threshold or qry_purity < purity_threshold:
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
                    'query_regime': int(qry_mode),
                    'support_purity': float(sup_purity),
                    'query_purity': float(qry_purity),
                    'support_row_ids': support_row_ids,
                    'query_row_ids': query_row_ids,
                    'n_support_days': len(support_dates),
                    'n_query_days': len(query_dates),
                    'n_support_points': len(support_row_ids),
                    'n_query_points': len(query_row_ids)
                }
                
                episodes.append(episode)
            
            return episodes, skipped
        
        train_episodes, train_skipped = create_episode_list(train_dates, 'train')
        val_episodes, val_skipped = create_episode_list(val_dates, 'val')
        test_episodes, test_skipped = create_episode_list(test_dates, 'test')
        
        all_episodes = train_episodes + val_episodes + test_episodes
        
        print(f"\n✓ 총 {len(all_episodes)} episodes 생성")
        print(f"  Train: {len(train_episodes)} (skipped: {train_skipped})")
        print(f"  Val:   {len(val_episodes)} (skipped: {val_skipped})")
        print(f"  Test:  {len(test_episodes)} (skipped: {test_skipped})")
        
        return {
            'episodes': all_episodes,
            'panel_with_regime': df,
            'stats': {
                'n_train': len(train_episodes),
                'n_val': len(val_episodes),
                'n_test': len(test_episodes),
                'train_skipped': train_skipped,
                'val_skipped': val_skipped,
                'test_skipped': test_skipped
            }
        }
    
    def save_results(self, k, daily_summary, episodes_data, kmeans, scaler, purity_threshold):
        """K별 결과 저장"""
        
        output_dir = os.path.join(self.output_base_dir, f'K{k}_tau{purity_threshold:.2f}')
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"K={k} 결과 저장: {output_dir}")
        print('='*80)
        
        # 1. Panel data (with regime)
        panel_path = os.path.join(output_dir, f'sp500_panel_K{k}.csv')
        episodes_data['panel_with_regime'].to_csv(panel_path)
        print(f"✓ Panel data: {panel_path}")
        
        # 2. Daily summary (with regime)
        summary_path = os.path.join(output_dir, f'market_regimes_K{k}.csv')
        daily_summary.to_csv(summary_path)
        print(f"✓ Market regimes: {summary_path}")
        
        # 3. Episodes
        episodes_path = os.path.join(output_dir, f'episodes_K{k}.json')
        with open(episodes_path, 'w') as f:
            json.dump(episodes_data['episodes'], f, indent=2)
        print(f"✓ Episodes: {episodes_path}")
        
        # 4. K-means model & scaler (pickle)
        import pickle
        model_path = os.path.join(output_dir, f'kmeans_model_K{k}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump({'kmeans': kmeans, 'scaler': scaler}, f)
        print(f"✓ K-means model: {model_path}")
        
        # 5. Metadata
        episodes_df = pd.DataFrame(episodes_data['episodes'])
        
        regime_dist = {}
        for split in ['train', 'val', 'test']:
            split_df = episodes_df[episodes_df['split'] == split]
            if len(split_df) > 0:
                regime_counts = split_df['support_regime'].value_counts().sort_index().to_dict()
                regime_dist[split] = regime_counts
        
        metadata = {
            'k': k,
            'n_regimes': k,
            'purity_threshold': float(purity_threshold),
            'support_len': 60,
            'query_len': 60,
            'train_only_clustering': True,  # 명시
            'train_end_date': str(self.train_end_date),
            'val_end_date': str(self.val_end_date),
            'n_features': len([col for col in episodes_data['panel_with_regime'].columns 
                              if col not in ['ticker', 'returns', 'row_id', 'regime']]),
            'episodes': episodes_data['stats'],
            'regime_distribution': regime_dist,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        metadata_path = os.path.join(output_dir, f'metadata_K{k}.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"✓ Metadata: {metadata_path}")
        
        # 6. Regime 특성 (Train 기준)
        train_mask = daily_summary.index <= self.train_end_date
        train_daily = daily_summary[train_mask]
        regime_chars = train_daily.groupby('regime')[
            ['market_return', 'market_volatility', 'vix', 'treasury_10y', 'yield_spread']
        ].mean()
        
        regime_chars_path = os.path.join(output_dir, f'regime_characteristics_K{k}.csv')
        regime_chars.to_csv(regime_chars_path)
        print(f"✓ Regime characteristics: {regime_chars_path}")
        
        return output_dir
    
    def process_k_value(self, k, purity_threshold=0.70):
        """단일 K값에 대한 전체 처리 (train-only clustering)"""
        print(f"\n{'='*80}")
        print(f"K={k} 처리 시작 (Train-Only Clustering)")
        print('='*80)
        
        # 1. Regime clustering (Train-only)
        daily_summary_with_regime, kmeans, scaler = self.cluster_regimes_train_only(k)
        
        # 2. Episode 생성
        episodes_data = self.create_episodes(
            k=k, 
            daily_summary=daily_summary_with_regime,
            purity_threshold=purity_threshold
        )
        
        # 3. 결과 저장
        output_dir = self.save_results(
            k,
            daily_summary_with_regime,
            episodes_data,
            kmeans,
            scaler,
            purity_threshold
        )

        return output_dir
    
    def batch_process(self, k_values=[3,4,5,6], purity_values=[0.50,0.55,0.60,0.65,0.75,0.80]):
        print("="*80)
        print("배치 전처리 시작 (Train-Only Regime Clustering)")
        print("="*80)
        print(f"\n처리할 K 값: {k_values}")
        print(f"Purity grid: {purity_values}")

        # 데이터 준비
        self.load_and_prepare_data()
        self.impute_with_train_only()
        self.compute_market_summary()

        results = {}

        # 🔥 K-τ 전체 그리드 실행
        for k in k_values:
            for tau in purity_values:
                try:
                    print(f"\n---- Running K={k}, τ={tau} ----")
                    output_dir = self.process_k_value(k, purity_threshold=tau)

                    results[(k, tau)] = {
                        'status': 'success',
                        'output_dir': output_dir
                    }

                except Exception as e:
                    print(f"\n❌ K={k}, τ={tau} 처리 실패: {e}")
                    results[(k, tau)] = {
                        'status': 'failed',
                        'error': str(e)
                    }

        # K-τ 비교 보고서 생성
        self.create_comparison_report(k_values, purity_values, results)
        return results

    
    def create_comparison_report(self, k_values, purity_values, results):

        rows = []
        for k in k_values:
            for tau in purity_values:
                key = (k, tau)
                if key not in results or results[key]['status'] != 'success':
                    continue

                meta_path = os.path.join(
                    results[key]['output_dir'], 
                    f'metadata_K{k}.json'
                )
                meta = json.load(open(meta_path))

                rows.append({
                    'K': k,
                    'tau': tau,
                    'Train Episodes': meta['episodes']['n_train'],
                    'Val Episodes': meta['episodes']['n_val'],
                    'Test Episodes': meta['episodes']['n_test'],
                })

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(self.output_base_dir, 'K_tau_comparison.csv'), index=False)
        print(df)
        return df



if __name__ == "__main__":
    processor = BatchPreprocessor(
        panel_path='./sp500_data/sp500_panel.csv',
        output_base_dir='./processed_data'
    )

    results = processor.batch_process(
        k_values=[3,4,5,6],
        purity_values=[0.50, 0.55, 0.60, 0.65, 0.75, 0.80]
    )
