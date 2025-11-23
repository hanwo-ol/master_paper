import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import json
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class SP500MetaLearningPreprocessor:
    """
    S&P 500 데이터 전처리 및 Meta-Learning Episode 생성
    
    논문의 실험 설계에 맞춘 전처리 파이프라인:
    - 39 features + 1 target (returns)
    - K=4 market regimes (k-means clustering)
    - Regime-pure episodes (support/query 동일 regime)
    - Time-ordered train/val/test split
    - N-asset subset experiments (N=10,50,100,200,400,all)
    """
    
    def __init__(self, data_dir='./sp500_data', output_dir='./processed_data'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.panel_data = None
        self.regimes = None
        self.episodes = None
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"Preprocessor 초기화 완료")
        print(f"  - Input: {data_dir}")
        print(f"  - Output: {output_dir}")
    
    def load_panel_data(self):
        """Panel 데이터 로드"""
        print("\n" + "="*80)
        print("Panel 데이터 로드 시작")
        print("="*80)
        
        panel_path = os.path.join(self.data_dir, 'sp500_panel.csv')
        
        # CSV 로드 (첫 번째 컬럼이 날짜 인덱스)
        self.panel_data = pd.read_csv(panel_path, index_col=0, parse_dates=True)
        
        print(f"  로드된 shape: {self.panel_data.shape}")
        print(f"  Index name: {self.panel_data.index.name}")
        print(f"  Columns: {list(self.panel_data.columns[:5])}...")
        
        # Row index 추가 (PyTorch Dataset용)
        # reset_index()로 날짜를 컬럼으로 변환
        self.panel_data = self.panel_data.reset_index()
        
        # 인덱스 컬럼명 확인 및 통일
        date_col = self.panel_data.columns[0]  # 첫 번째 컬럼이 날짜
        print(f"  Date column detected: '{date_col}'")
        
        # 날짜 컬럼명을 'date'로 통일
        if date_col != 'date':
            self.panel_data = self.panel_data.rename(columns={date_col: 'date'})
            print(f"  Renamed '{date_col}' → 'date'")
        
        # 날짜 타입 확인
        self.panel_data['date'] = pd.to_datetime(self.panel_data['date'])
        
        # Row ID 추가
        self.panel_data['row_id'] = np.arange(len(self.panel_data))
        
        # 날짜를 다시 인덱스로 설정
        self.panel_data = self.panel_data.set_index('date')
        
        print(f"\n✓ Panel 데이터 로드 완료")
        print(f"  - Shape: {self.panel_data.shape}")
        print(f"  - Index: {self.panel_data.index.name} (dtype: {self.panel_data.index.dtype})")
        print(f"  - Columns: {len(self.panel_data.columns)}")
        print(f"  - Features: 39 (+ returns + ticker + row_id)")
        print(f"  - Memory: {self.panel_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        return self.panel_data

    
    def eda_summary(self):
        """패널 데이터 EDA (간략)"""
        print("\n" + "="*80)
        print("패널 데이터 EDA")
        print("="*80)
        
        df = self.panel_data
        
        # 1. 기본 정보
        print("\n[1] 기본 정보")
        print(f"  - 총 데이터 포인트: {len(df):,}")
        print(f"  - 유니크 티커 수: {df['ticker'].nunique()}")
        print(f"  - 날짜 범위: {df.index.min()} ~ {df.index.max()}")
        print(f"  - 총 기간: {(df.index.max() - df.index.min()).days} days (~{(df.index.max() - df.index.min()).days/252:.1f} years)")
        
        # 2. Missing 비율
        print("\n[2] Missing Values")
        feature_cols = [col for col in df.columns if col not in ['ticker', 'returns', 'row_id']]
        missing_ratio = (df[feature_cols].isnull().sum() / len(df) * 100).sort_values(ascending=False)
        high_missing = missing_ratio[missing_ratio > 5]
        
        if len(high_missing) > 0:
            print(f"  ⚠ Missing > 5%인 컬럼 ({len(high_missing)}개):")
            for col, ratio in high_missing.head(10).items():
                print(f"    - {col}: {ratio:.2f}%")
        else:
            print(f"  ✓ 모든 feature의 missing < 5%")
        
        print(f"  - 전체 평균 missing: {missing_ratio.mean():.3f}%")
        
        # 3. Extreme 값 체크
        print("\n[3] Extreme Values 체크 (±5 std)")
        extreme_counts = {}
        for col in feature_cols:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                extreme = ((df[col] < mean - 5*std) | (df[col] > mean + 5*std)).sum()
                if extreme > 0:
                    extreme_counts[col] = extreme
        
        if extreme_counts:
            total_extremes = sum(extreme_counts.values())
            print(f"  ⚠ Extreme 값 총 {total_extremes:,}개 ({total_extremes/len(df)/len(feature_cols)*100:.3f}%)")
            print(f"    영향받는 컬럼: {len(extreme_counts)}개")
            top_extreme = sorted(extreme_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            for col, count in top_extreme:
                print(f"      - {col}: {count:,}")
        else:
            print(f"  ✓ Extreme 값 없음")
        
        # 4. Returns 분포
        print("\n[4] Returns 분포")
        returns = df['returns'].dropna()
        print(f"  - Mean: {returns.mean():.6f} ({returns.mean()*252:.2%} annualized)")
        print(f"  - Std: {returns.std():.6f} ({returns.std()*np.sqrt(252):.2%} annualized)")
        print(f"  - Skewness: {returns.skew():.3f}")
        print(f"  - Kurtosis: {returns.kurtosis():.3f}")
        print(f"  - Min/Max: [{returns.min():.3f}, {returns.max():.3f}]")
        
        # 5. 티커별 분포
        print("\n[5] 티커별 데이터 분포")
        ticker_counts = df.groupby('ticker').size()
        print(f"  - 평균 관측치/티커: {ticker_counts.mean():.0f}")
        print(f"  - 중앙값: {ticker_counts.median():.0f}")
        print(f"  - 최소/최대: [{ticker_counts.min()}, {ticker_counts.max()}]")
        
        return {
            'total_points': len(df),
            'n_tickers': df['ticker'].nunique(),
            'date_range': (df.index.min(), df.index.max()),
            'missing_ratio': missing_ratio.to_dict(),
            'extreme_counts': extreme_counts
        }
    
    def compute_market_summary(self):
        """Daily market summary 계산 (regime 라벨링용)"""
        print("\n" + "="*80)
        print("Market Summary 계산 (Regime Clustering)")
        print("="*80)
        
        df = self.panel_data
        
        # 날짜별로 집계
        print("\n일별 시장 지표 계산 중...")
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
        
        # 컬럼명 정리
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
        
        # 추가 지표
        daily_summary['volume_change'] = daily_summary['total_volume'].pct_change()
        daily_summary['vix_change'] = daily_summary['vix'].pct_change()
        
        # 이동평균 (regime 특징화)
        for window in [5, 20]:
            daily_summary[f'ma_return_{window}d'] = daily_summary['market_return'].rolling(window).mean()
            daily_summary[f'ma_vol_{window}d'] = daily_summary['market_volatility'].rolling(window).mean()
        
        daily_summary = daily_summary.dropna()
        
        print(f"✓ Daily summary 계산 완료: {daily_summary.shape}")
        
        return daily_summary
    
    def label_regimes(self, n_regimes=4, features=None, random_state=42):
        """
        K-means를 사용한 Market Regime 라벨링
        
        Args:
            n_regimes: Regime 개수 (K). 논문에서는 K=4 사용
            features: Clustering에 사용할 feature 리스트
            random_state: 재현성을 위한 seed
        """
        print("\n" + "="*80)
        print(f"Market Regime 라벨링 (K={n_regimes})")
        print("="*80)
        
        # Market summary 계산
        daily_summary = self.compute_market_summary()
        
        # Clustering features
        if features is None:
            features = [
                'market_return', 'market_volatility', 'vix',
                'treasury_10y', 'yield_spread',
                'ma_return_5d', 'ma_vol_5d',
                'ma_return_20d', 'ma_vol_20d'
            ]
        
        print(f"\nClustering features ({len(features)}개):")
        for f in features:
            print(f"  - {f}")
        
        X = daily_summary[features].values
        
        # 표준화
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # K-means
        print(f"\nK-means 학습 중 (K={n_regimes}, seed={random_state})...")
        kmeans = KMeans(n_clusters=n_regimes, random_state=random_state, n_init=20)
        regime_labels = kmeans.fit_predict(X_scaled)
        
        daily_summary['regime'] = regime_labels
        
        # Regime 통계
        print(f"\n✓ Regime 라벨링 완료")
        print(f"\nRegime 분포:")
        regime_counts = pd.Series(regime_labels).value_counts().sort_index()
        for regime_id, count in regime_counts.items():
            pct = count / len(regime_labels) * 100
            print(f"  Regime {regime_id}: {count:,} days ({pct:.1f}%)")
        
        # Regime 특성
        print(f"\nRegime별 평균 특성:")
        regime_chars = daily_summary.groupby('regime')[['market_return', 'market_volatility', 
                                                         'vix', 'treasury_10y', 'yield_spread']].mean()
        print(regime_chars.round(4))
        
        # Panel에 병합
        print(f"\nPanel 데이터에 regime 병합 중...")
        self.panel_data = self.panel_data.join(daily_summary[['regime']], how='left')
        self.regimes = daily_summary
        
        print(f"✓ Regime 병합 완료")
        
        # 저장
        regime_path = os.path.join(self.output_dir, 'market_regimes.csv')
        daily_summary.to_csv(regime_path)
        
        # Regime metadata
        regime_metadata = {
            'n_regimes': n_regimes,
            'clustering_features': features,
            'random_state': random_state,
            'regime_distribution': regime_counts.to_dict(),
            'regime_characteristics': regime_chars.to_dict('index')
        }
        
        metadata_path = os.path.join(self.output_dir, 'regime_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(regime_metadata, f, indent=2, default=str)
        
        print(f"\n✓ Regime 데이터 저장: {regime_path}")
        print(f"✓ Metadata 저장: {metadata_path}")
        
        return daily_summary
    
    def create_episodes(self, support_len=60, query_len=60, 
                       train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                       min_history=252, purity_threshold=0.8, 
                       regime_pure=True):
        """
        Meta-Learning Episodes 생성
        
        Args:
            support_len: Support set 길이 (days)
            query_len: Query set 길이 (days)
            train_ratio: Training set 비율
            val_ratio: Validation set 비율
            test_ratio: Test set 비율
            min_history: 최소 이력 (warm-up)
            purity_threshold: Regime purity 임계값 (0.8 = 80%)
            regime_pure: True면 support/query가 같은 regime이고 purity > threshold인 것만
        
        Returns:
            episodes: List of episode dictionaries
        """
        print("\n" + "="*80)
        print("Meta-Learning Episodes 생성")
        print("="*80)
        
        print(f"\n설정:")
        print(f"  - Support length: {support_len} days")
        print(f"  - Query length: {query_len} days")
        print(f"  - Episode length: {support_len + query_len} days")
        print(f"  - Min history: {min_history} days (warm-up)")
        print(f"  - Train/Val/Test: {train_ratio}/{val_ratio}/{test_ratio}")
        print(f"  - Regime-pure: {regime_pure}")
        if regime_pure:
            print(f"  - Purity threshold: {purity_threshold} ({purity_threshold*100}%)")
        
        df = self.panel_data
        
        # 날짜 리스트
        unique_dates = sorted(df.index.unique())
        print(f"\n총 거래일: {len(unique_dates)} days")
        
        # Episode 시작 가능 날짜
        start_idx = min_history + support_len
        end_idx = len(unique_dates) - query_len
        
        available_starts = unique_dates[start_idx:end_idx]
        print(f"Episode 생성 가능 날짜: {len(available_starts)} days")
        print(f"  첫 가능 날짜: {available_starts[0]}")
        print(f"  마지막 가능 날짜: {available_starts[-1]}")
        
        # Train/Val/Test split (시간 순서 유지)
        n_total = len(available_starts)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val
        
        train_dates = available_starts[:n_train]
        val_dates = available_starts[n_train:n_train+n_val]
        test_dates = available_starts[n_train+n_val:]
        
        print(f"\nSplit 결과:")
        print(f"  Train: {len(train_dates)} slots ({train_dates[0]} ~ {train_dates[-1]})")
        print(f"  Val:   {len(val_dates)} slots ({val_dates[0]} ~ {val_dates[-1]})")
        print(f"  Test:  {len(test_dates)} slots ({test_dates[0]} ~ {test_dates[-1]})")
        
        # Episodes 생성
        def create_episode_list(date_list, split_name):
            episodes = []
            skipped = 0
            
            for start_date in tqdm(date_list, desc=f"{split_name} episodes"):
                # Support set
                start_idx_local = unique_dates.index(start_date) - support_len
                support_dates = unique_dates[start_idx_local:start_idx_local + support_len]
                
                # Query set
                query_start_idx = start_idx_local + support_len
                query_dates = unique_dates[query_start_idx:query_start_idx + query_len]
                
                # Regime 정보
                support_mask = df.index.isin(support_dates)
                query_mask = df.index.isin(query_dates)
                
                support_regimes = df.loc[support_mask, 'regime'].dropna()
                query_regimes = df.loc[query_mask, 'regime'].dropna()
                
                if len(support_regimes) == 0 or len(query_regimes) == 0:
                    skipped += 1
                    continue
                
                sup_mode = support_regimes.mode()[0]
                qry_mode = query_regimes.mode()[0]
                
                # Regime-pure 조건 체크
                if regime_pure:
                    sup_purity = (support_regimes == sup_mode).mean()
                    qry_purity = (query_regimes == qry_mode).mean()
                    
                    # Support와 Query가 다른 regime이거나 purity가 낮으면 제외
                    if sup_mode != qry_mode:
                        skipped += 1
                        continue
                    if sup_purity < purity_threshold or qry_purity < purity_threshold:
                        skipped += 1
                        continue
                
                # Row indices (PyTorch Dataset용)
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
        
        print("\nEpisode 리스트 생성 중...")
        train_episodes, train_skipped = create_episode_list(train_dates, 'train')
        val_episodes, val_skipped = create_episode_list(val_dates, 'val')
        test_episodes, test_skipped = create_episode_list(test_dates, 'test')
        
        all_episodes = train_episodes + val_episodes + test_episodes
        
        print(f"\n✓ 총 {len(all_episodes)} episodes 생성 완료")
        print(f"  - Train: {len(train_episodes)} (skipped: {train_skipped})")
        print(f"  - Val: {len(val_episodes)} (skipped: {val_skipped})")
        print(f"  - Test: {len(test_episodes)} (skipped: {test_skipped})")
        
        # Regime 분포
        print(f"\nRegime 분포 (support set 기준):")
        episodes_df = pd.DataFrame(all_episodes)
        for split in ['train', 'val', 'test']:
            split_df = episodes_df[episodes_df['split'] == split]
            regime_dist = split_df['support_regime'].value_counts().sort_index()
            print(f"  {split.capitalize()}:")
            for regime, count in regime_dist.items():
                pct = count / len(split_df) * 100
                print(f"    Regime {regime}: {count} ({pct:.1f}%)")
        
        # 저장
        episodes_path = os.path.join(self.output_dir, 'episodes.json')
        with open(episodes_path, 'w') as f:
            json.dump(all_episodes, f, indent=2)
        print(f"\n✓ Episodes 저장: {episodes_path}")
        
        # Metadata
        metadata = {
            'support_len': support_len,
            'query_len': query_len,
            'episode_len': support_len + query_len,
            'min_history': min_history,
            'regime_pure': regime_pure,
            'purity_threshold': purity_threshold if regime_pure else None,
            'n_train': len(train_episodes),
            'n_val': len(val_episodes),
            'n_test': len(test_episodes),
            'n_total': len(all_episodes),
            'n_skipped': {
                'train': train_skipped,
                'val': val_skipped,
                'test': test_skipped
            },
            'date_range': {
                'train': (str(train_dates[0]), str(train_dates[-1])),
                'val': (str(val_dates[0]), str(val_dates[-1])),
                'test': (str(test_dates[0]), str(test_dates[-1]))
            }
        }
        
        metadata_path = os.path.join(self.output_dir, 'episodes_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata 저장: {metadata_path}")
        
        self.episodes = all_episodes
        return all_episodes
    
    def create_ticker_subsets(self, n_assets_list=[10, 50, 100, 200, 400], seed=42):
        """
        N=10, 50, 100, 200, 400 실험을 위한 ticker subset 생성
        
        동일 episode 구조를 유지하되, ticker만 subset으로 필터링
        """
        print("\n" + "="*80)
        print("Ticker Subsets 생성 (N-asset experiments)")
        print("="*80)
        
        all_tickers = sorted(self.panel_data['ticker'].unique())
        print(f"\n전체 티커 수: {len(all_tickers)}")
        
        np.random.seed(seed)
        
        subsets = {}
        for n in n_assets_list:
            if n > len(all_tickers):
                print(f"  ⚠ N={n}은 전체 티커 수({len(all_tickers)})를 초과합니다. 스킵.")
                continue
            
            # 랜덤 샘플링
            selected = np.random.choice(all_tickers, size=n, replace=False).tolist()
            subsets[f'N{n}'] = sorted(selected)
            print(f"  ✓ N={n}: {len(selected)}개 티커 선택")
        
        # 전체 세트
        subsets['N_all'] = all_tickers
        print(f"  ✓ N=all: {len(all_tickers)}개 티커 (전체)")
        
        # 저장
        subsets_path = os.path.join(self.output_dir, 'ticker_subsets.json')
        with open(subsets_path, 'w') as f:
            json.dump(subsets, f, indent=2)
        print(f"\n✓ Ticker subsets 저장: {subsets_path}")
        
        # Subset 메타데이터
        subset_metadata = {
            'seed': seed,
            'n_assets_list': n_assets_list,
            'total_tickers': len(all_tickers),
            'subset_sizes': {k: len(v) for k, v in subsets.items()}
        }
        
        metadata_path = os.path.join(self.output_dir, 'ticker_subsets_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(subset_metadata, f, indent=2)
        print(f"✓ Subset metadata 저장: {metadata_path}")
        
        return subsets
    
    def save_processed_panel(self):
        """전처리된 panel 데이터 저장"""
        print("\n" + "="*80)
        print("전처리된 Panel 데이터 저장")
        print("="*80)
        
        output_path = os.path.join(self.output_dir, 'sp500_panel_processed.csv')
        self.panel_data.to_csv(output_path)
        
        print(f"✓ 저장 완료: {output_path}")
        print(f"  - Shape: {self.panel_data.shape}")
        print(f"  - Columns: {list(self.panel_data.columns)}")
        print(f"  - Size: {os.path.getsize(output_path) / 1024**2:.1f} MB")
        
        return output_path
    
    def generate_summary_report(self):
        """전처리 요약 리포트 생성"""
        print("\n" + "="*80)
        print("전처리 요약 리포트")
        print("="*80)
        
        report = {
            'dataset': {
                'n_tickers': self.panel_data['ticker'].nunique(),
                'n_total_points': len(self.panel_data),
                'date_range': (str(self.panel_data.index.min()), str(self.panel_data.index.max())),
                'n_features': 39,
                'n_days': len(self.panel_data.index.unique())
            },
            'regimes': {
                'n_regimes': int(self.panel_data['regime'].nunique()),
                'distribution': self.panel_data.groupby('regime').size().to_dict()
            },
            'episodes': {
                'n_train': len([e for e in self.episodes if e['split'] == 'train']),
                'n_val': len([e for e in self.episodes if e['split'] == 'val']),
                'n_test': len([e for e in self.episodes if e['split'] == 'test']),
                'n_total': len(self.episodes)
            }
        }
        
        report_path = os.path.join(self.output_dir, 'preprocessing_summary.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n✓ 요약 리포트 저장: {report_path}")
        print(f"\n최종 데이터셋:")
        print(f"  - {report['dataset']['n_tickers']} tickers")
        print(f"  - {report['dataset']['n_total_points']:,} data points")
        print(f"  - {report['dataset']['n_features']} features")
        print(f"  - {report['regimes']['n_regimes']} regimes")
        print(f"  - {report['episodes']['n_total']} episodes")
        
        return report


# 실행 예제
if __name__ == "__main__":
    print("S&P 500 Meta-Learning Preprocessing Pipeline (Final Version)")
    print("=" * 80)
    print("\n논문 실험 설계:")
    print("  - 39 features + 1 target (returns)")
    print("  - K=4 market regimes")
    print("  - Regime-pure episodes (purity ≥ 80%)")
    print("  - Support/Query: 60/60 days")
    print("  - Train/Val/Test: 70/15/15 (time-ordered)")
    print("  - N-asset subsets: 10, 50, 100, 200, 400, all")
    
    # 1. Preprocessor 초기화
    preprocessor = SP500MetaLearningPreprocessor(
        data_dir='./sp500_data',
        output_dir='./processed_data'
    )
    
    # 2. Panel 데이터 로드
    panel = preprocessor.load_panel_data()
    
    # 3. EDA
    eda_results = preprocessor.eda_summary()
    
    # 4. Regime 라벨링 (K=4, 논문과 일치)
    regimes = preprocessor.label_regimes(n_regimes=4, random_state=42)
    
    # 5. Episodes 생성 (regime-pure, purity=0.8)
    episodes = preprocessor.create_episodes(
        support_len=60,
        query_len=60,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        min_history=252,
        purity_threshold=0.8,
        regime_pure=True  # 논문의 이론과 일치
    )
    
    # 6. Ticker subsets 생성
    subsets = preprocessor.create_ticker_subsets(
        n_assets_list=[10, 50, 100, 200, 400],
        seed=42
    )
    
    # 7. 전처리된 데이터 저장
    preprocessor.save_processed_panel()
    
    # 8. 요약 리포트
    report = preprocessor.generate_summary_report()
    
    print("\n" + "=" * 80)
    print("전처리 완료!")
    print("=" * 80)
    print("\n생성된 파일:")
    print("  📊 processed_data/sp500_panel_processed.csv       (전처리된 패널 데이터)")
    print("  🎯 processed_data/market_regimes.csv              (일별 regime 라벨)")
    print("  📝 processed_data/regime_metadata.json            (Regime 메타데이터)")
    print("  📚 processed_data/episodes.json                   (Episode 리스트)")
    print("  📋 processed_data/episodes_metadata.json          (Episode 메타데이터)")
    print("  🎲 processed_data/ticker_subsets.json             (N-asset subsets)")
    print("  📊 processed_data/ticker_subsets_metadata.json    (Subset 메타데이터)")
    print("  📄 processed_data/preprocessing_summary.json      (전체 요약)")
    
    print("\n다음 단계:")
    print("  ✅ PyTorch Dataset 구현")
    print("  ✅ Meta-learner (MAML/Reptile) 구현")
    print("  ✅ Baseline 모델 (Markowitz, EW, RP) 구현")
    print("  ✅ 실험 실행 및 결과 분석")
