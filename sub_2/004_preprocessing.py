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
    K=4,5,6에 대한 배치 전처리 및 Episode 생성
    각 K별로 독립적인 디렉토리에 결과 저장
    """
    
    def __init__(self, panel_path='./sp500_data/sp500_panel.csv', 
                 output_base_dir='./processed_data'):
        self.panel_path = panel_path
        self.output_base_dir = output_base_dir
        self.panel_data = None
        self.train_end_date = '2019-10-28'  # Train/Val split
        
        os.makedirs(output_base_dir, exist_ok=True)
    
    def load_and_prepare_data(self):
        """데이터 로드 및 기본 전처리"""
        print("="*80)
        print("데이터 로드 및 전처리")
        print("="*80)
        
        # 로드
        df = pd.read_csv(self.panel_path, index_col=0, parse_dates=True)
        print(f"\n원본 shape: {df.shape}")
        
        # 1. asset_turnover 제거
        if 'asset_turnover' in df.columns:
            df = df.drop(columns=['asset_turnover'])
            print("✓ asset_turnover 제거")
        
        # 2. Missing indicator 추가
        fundamental_cols = ['pe_ratio', 'pb_ratio', 'roe', 'roa', 'debt_to_equity',
                           'dividend_yield', 'payout_ratio', 'free_cashflow_yield']
        fundamental_cols = [col for col in fundamental_cols if col in df.columns]
        
        print(f"\n✓ Missing indicators 추가 ({len(fundamental_cols)}개)")
        for col in fundamental_cols:
            df[f'{col}_missing'] = df[col].isnull().astype(int)
        
        # 3. Train-only imputation
        train_mask = df.index <= self.train_end_date
        train_df = df[train_mask]
        
        print(f"\n✓ Train-only imputation (Train end: {self.train_end_date})")
        
        # Fundamental (train median)
        for col in fundamental_cols:
            train_median = train_df[col].median()
            df[col] = df.groupby('ticker')[col].transform(lambda x: x.fillna(method='ffill'))
            df[col] = df[col].fillna(train_median)
        
        # Macro (forward fill)
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
        
        # Row ID 추가
        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: 'date'})
        df['date'] = pd.to_datetime(df['date'])
        df['row_id'] = np.arange(len(df))
        df = df.set_index('date')
        
        final_missing = df.isnull().sum().sum()
        feature_count = len([col for col in df.columns if col not in ['ticker', 'returns', 'row_id']])
        
        print(f"\n✓ 최종 missing: {final_missing:,}")
        print(f"✓ 최종 feature 수: {feature_count}")
        
        self.panel_data = df
        return df
    
    def compute_market_summary(self):
        """Market summary 계산"""
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
        
        return daily_summary
    
    def cluster_regimes(self, k, daily_summary):
        """K-means clustering으로 regime 라벨링"""
        features = [
            'market_return', 'market_volatility', 'vix',
            'treasury_10y', 'yield_spread',
            'ma_return_5d', 'ma_vol_5d',
            'ma_return_20d', 'ma_vol_20d'
        ]
        
        X = daily_summary[features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        regime_labels = kmeans.fit_predict(X_scaled)
        
        daily_summary_copy = daily_summary.copy()
        daily_summary_copy['regime'] = regime_labels
        
        return daily_summary_copy, kmeans
    
    def create_episodes(self, k, daily_summary, purity_threshold=0.70,
                       support_len=60, query_len=60):
        """Episode 생성 (regime-pure)"""
        
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
        
        # Train/Val/Test split
        n_total = len(available_starts)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.15)
        
        train_dates = available_starts[:n_train]
        val_dates = available_starts[n_train:n_train+n_val]
        test_dates = available_starts[n_train+n_val:]
        
        def create_episode_list(date_list, split_name):
            episodes = []
            skipped = 0
            
            for start_date in tqdm(date_list, desc=f"{split_name} episodes (K={k})"):
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
    
    def save_results(self, k, daily_summary, episodes_data, kmeans):
        """K별 결과 저장"""
        
        # 출력 디렉토리 생성
        output_dir = os.path.join(self.output_base_dir, f'K{k}')
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
        
        # 4. Metadata
        episodes_df = pd.DataFrame(episodes_data['episodes'])
        
        regime_dist = {}
        for split in ['train', 'val', 'test']:
            split_df = episodes_df[episodes_df['split'] == split]
            regime_counts = split_df['support_regime'].value_counts().sort_index().to_dict()
            regime_dist[split] = regime_counts
        
        metadata = {
            'k': k,
            'n_regimes': k,
            'purity_threshold': 0.70,
            'support_len': 60,
            'query_len': 60,
            'n_features': len([col for col in episodes_data['panel_with_regime'].columns 
                              if col not in ['ticker', 'returns', 'row_id', 'regime']]),
            'episodes': episodes_data['stats'],
            'regime_distribution': regime_dist,
            'date_range': {
                'train': (episodes_data['episodes'][0]['support_start'],
                         episodes_df[episodes_df['split']=='train'].iloc[-1]['query_end']),
                'val': (episodes_df[episodes_df['split']=='val'].iloc[0]['support_start'],
                       episodes_df[episodes_df['split']=='val'].iloc[-1]['query_end']),
                'test': (episodes_df[episodes_df['split']=='test'].iloc[0]['support_start'],
                        episodes_df[episodes_df['split']=='test'].iloc[-1]['query_end'])
            },
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        metadata_path = os.path.join(output_dir, f'metadata_K{k}.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"✓ Metadata: {metadata_path}")
        
        # 5. Regime 특성 분석
        regime_chars = daily_summary.groupby('regime')[
            ['market_return', 'market_volatility', 'vix', 'treasury_10y', 'yield_spread']
        ].mean()
        
        regime_chars_path = os.path.join(output_dir, f'regime_characteristics_K{k}.csv')
        regime_chars.to_csv(regime_chars_path)
        print(f"✓ Regime characteristics: {regime_chars_path}")
        
        # 6. 요약 출력
        print(f"\n[K={k} 요약]")
        print(f"  Episodes: Train={episodes_data['stats']['n_train']}, "
              f"Val={episodes_data['stats']['n_val']}, "
              f"Test={episodes_data['stats']['n_test']}")
        print(f"  Regime 분포:")
        for regime_id in range(k):
            n_days = (daily_summary['regime'] == regime_id).sum()
            pct = n_days / len(daily_summary) * 100
            print(f"    Regime {regime_id}: {n_days} days ({pct:.1f}%)")
        
        return output_dir
    
    def process_k_value(self, k, purity_threshold=0.70):
        """단일 K값에 대한 전체 처리"""
        print(f"\n{'='*80}")
        print(f"K={k} 처리 시작")
        print('='*80)
        
        # 1. Market summary 계산
        daily_summary = self.compute_market_summary()
        
        # 2. Regime clustering
        daily_summary_with_regime, kmeans = self.cluster_regimes(k, daily_summary)
        
        # 3. Episode 생성
        episodes_data = self.create_episodes(
            k=k, 
            daily_summary=daily_summary_with_regime,
            purity_threshold=purity_threshold
        )
        
        # 4. 결과 저장
        output_dir = self.save_results(k, daily_summary_with_regime, episodes_data, kmeans)
        
        return output_dir
    
    def batch_process(self, k_values=[4, 5, 6], purity_threshold=0.70):
        """배치 처리: K=4,5,6에 대해 각각 실행"""
        print("="*80)
        print("배치 전처리 시작")
        print("="*80)
        print(f"\n처리할 K 값: {k_values}")
        print(f"Purity threshold: {purity_threshold}")
        
        # 데이터 준비 (한 번만)
        self.load_and_prepare_data()
        
        # K별 처리
        results = {}
        for k in k_values:
            try:
                output_dir = self.process_k_value(k, purity_threshold)
                results[k] = {
                    'status': 'success',
                    'output_dir': output_dir
                }
            except Exception as e:
                print(f"\n❌ K={k} 처리 실패: {e}")
                results[k] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        # 전체 요약
        print("\n" + "="*80)
        print("배치 처리 완료")
        print("="*80)
        
        for k, result in results.items():
            if result['status'] == 'success':
                print(f"✓ K={k}: {result['output_dir']}")
            else:
                print(f"❌ K={k}: {result['error']}")
        
        # 비교 리포트 생성
        self.create_comparison_report(k_values, results)
        
        return results
    
    def create_comparison_report(self, k_values, results):
        """K별 비교 리포트 생성"""
        print("\n" + "="*80)
        print("K별 비교 리포트")
        print("="*80)
        
        comparison = []
        
        for k in k_values:
            if results[k]['status'] != 'success':
                continue
            
            metadata_path = os.path.join(results[k]['output_dir'], f'metadata_K{k}.json')
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            comparison.append({
                'K': k,
                'Train Episodes': metadata['episodes']['n_train'],
                'Val Episodes': metadata['episodes']['n_val'],
                'Test Episodes': metadata['episodes']['n_test'],
                'Total Episodes': (metadata['episodes']['n_train'] + 
                                  metadata['episodes']['n_val'] + 
                                  metadata['episodes']['n_test']),
                'Train Skipped': metadata['episodes']['train_skipped'],
                'Val Skipped': metadata['episodes']['val_skipped'],
                'Test Skipped': metadata['episodes']['test_skipped']
            })
        
        comparison_df = pd.DataFrame(comparison)
        
        print("\n")
        print(comparison_df.to_string(index=False))
        
        # 저장
        comparison_path = os.path.join(self.output_base_dir, 'K_comparison.csv')
        comparison_df.to_csv(comparison_path, index=False)
        print(f"\n✓ 비교 리포트 저장: {comparison_path}")
        
        return comparison_df


# 실행
if __name__ == "__main__":
    print("S&P 500 Batch Preprocessing (K=4,5,6)")
    print("="*80)
    
    preprocessor = BatchPreprocessor(
        panel_path='./sp500_data/sp500_panel.csv',
        output_base_dir='./processed_data'
    )
    
    # K=4,5,6에 대해 배치 처리
    results = preprocessor.batch_process(
        k_values=[3, 4, 5, 6],
        purity_threshold=0.70
    )
    
    print("\n" + "="*80)
    print("전체 처리 완료!")
    print("="*80)
    print("\n생성된 디렉토리 구조:")
    print("  processed_data/")
    print("    ├── K3/")
    print("    │   └── (동일 구조)")
    print("    ├── K4/")
    print("    │   ├── sp500_panel_K4.csv")
    print("    │   ├── market_regimes_K4.csv")
    print("    │   ├── episodes_K4.json")
    print("    │   ├── metadata_K4.json")
    print("    │   └── regime_characteristics_K4.csv")
    print("    ├── K5/")
    print("    │   └── (동일 구조)")
    print("    ├── K6/")
    print("    │   └── (동일 구조)")
    print("    └── K_comparison.csv")
    
    print("\n다음 단계:")
    print("  1. K_comparison.csv로 최적 K 선택")
    print("  2. 선택한 K의 데이터로 PyTorch Dataset 구현")
    print("  3. Meta-learner 학습")
