#002_preprocessing.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import json
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class PreprocessingOptimizer:
    """
    전처리 최적화 분석:
    1. Feature selection (asset_turnover 제거 등)
    2. Optimal K (regimes) 선택
    3. Purity threshold 최적화
    """
    
    def __init__(self, panel_path='./sp500_data/sp500_panel.csv'):
        self.panel_path = panel_path
        self.panel_data = None
        self.daily_summary = None
        
    def load_and_clean(self):
        """데이터 로드 및 기본 정제"""
        print("="*80)
        print("데이터 로드 및 정제")
        print("="*80)
        
        # 로드
        df = pd.read_csv(self.panel_path, index_col=0, parse_dates=True)
        print(f"\n원본 shape: {df.shape}")
        
        # 1. Feature 분석
        print("\n[1] Feature Missing 분석:")
        feature_cols = [col for col in df.columns if col not in ['ticker', 'returns']]
        missing_pct = (df[feature_cols].isnull().sum() / len(df) * 100).sort_values(ascending=False)
        
        print("\nMissing > 50%:")
        high_missing = missing_pct[missing_pct > 50]
        for col, pct in high_missing.items():
            print(f"  ❌ {col}: {pct:.1f}%")
        
        print("\nMissing 10-50%:")
        medium_missing = missing_pct[(missing_pct >= 10) & (missing_pct <= 50)]
        for col, pct in medium_missing.items():
            print(f"  ⚠️ {col}: {pct:.1f}%")
        
        # 2. asset_turnover 제거 결정
        if 'asset_turnover' in df.columns:
            turnover_missing = missing_pct.get('asset_turnover', 0)
            if turnover_missing > 90:
                print(f"\n✓ asset_turnover 제거 결정 (missing: {turnover_missing:.1f}%)")
                df = df.drop(columns=['asset_turnover'])
            else:
                print(f"\n⚠ asset_turnover 유지 (missing: {turnover_missing:.1f}%)")
        
        # 3. Missing value imputation 강화
        print("\n[2] Missing Value Imputation:")
        
        # Fundamental features (ticker별 forward fill → median)
        fundamental_cols = ['pe_ratio', 'pb_ratio', 'roe', 'roa', 'debt_to_equity',
                        'dividend_yield', 'payout_ratio', 'free_cashflow_yield']
        fundamental_cols = [col for col in fundamental_cols if col in df.columns]
        
        for col in fundamental_cols:
            before_missing = df[col].isnull().sum()
            
            # Ticker별 forward fill (transform 사용)
            df[col] = df.groupby('ticker')[col].transform(lambda x: x.fillna(method='ffill'))
            
            # 전체 median으로 나머지 채우기
            df[col] = df[col].fillna(df[col].median())
            
            after_missing = df[col].isnull().sum()
            print(f"  {col}: {before_missing:,} → {after_missing:,}")
        
        # Macro features (전체 forward fill)
        macro_cols = ['vix', 'treasury_10y', 'treasury_2y', 'yield_spread',
                    'usd_index', 'credit_spread', 'cpi_yoy']
        macro_cols = [col for col in macro_cols if col in df.columns]
        
        for col in macro_cols:
            before_missing = df[col].isnull().sum()
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
            after_missing = df[col].isnull().sum()
            if before_missing > 0:
                print(f"  {col}: {before_missing:,} → {after_missing:,}")
        
        # Technical indicators (ticker별 forward fill → backfill)
        tech_cols = ['rsi', 'ma_20', 'ma_50', 'ma_200', 'macd', 'macd_signal',
                    'bollinger_upper', 'bollinger_lower', 'atr', 'stoch_k', 'stoch_d',
                    'price_roc', 'realized_vol_20', 'realized_vol_60', 'volume_roc',
                    'mfi', 'obv', 'williams_r']
        tech_cols = [col for col in tech_cols if col in df.columns]
        
        for col in tech_cols:
            # Transform 사용
            df[col] = df.groupby('ticker')[col].transform(
                lambda x: x.fillna(method='ffill').fillna(method='bfill')
            )
        
        # 최종 missing 확인
        final_missing = df.isnull().sum().sum()
        print(f"\n✓ 최종 missing values: {final_missing:,}")
        
        # 최종 feature 수
        final_features = [col for col in df.columns if col not in ['ticker', 'returns']]
        print(f"✓ 최종 feature 수: {len(final_features)}")
        
        # Feature 리스트 출력
        print(f"\n최종 Feature 리스트:")
        price_features = ['open', 'high', 'low', 'close', 'volume']
        tech_features = [f for f in final_features if f in tech_cols]
        fund_features = [f for f in final_features if f in fundamental_cols]
        macro_features = [f for f in final_features if f in macro_cols]
        
        print(f"  - Price (5): {price_features}")
        print(f"  - Technical ({len(tech_features)}): {tech_features}")
        print(f"  - Fundamental ({len(fund_features)}): {fund_features}")
        print(f"  - Macro ({len(macro_features)}): {macro_features}")
        print(f"  - Total: {len(final_features)} features")
        
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
        
        # 추가 지표
        daily_summary['volume_change'] = daily_summary['total_volume'].pct_change()
        daily_summary['vix_change'] = daily_summary['vix'].pct_change()
        
        for window in [5, 20]:
            daily_summary[f'ma_return_{window}d'] = daily_summary['market_return'].rolling(window).mean()
            daily_summary[f'ma_vol_{window}d'] = daily_summary['market_volatility'].rolling(window).mean()
        
        daily_summary = daily_summary.dropna()
        
        print(f"✓ Daily summary: {daily_summary.shape}")
        
        self.daily_summary = daily_summary
        return daily_summary
    
    def find_optimal_k(self, k_range=range(2, 11), features=None):
        """
        최적의 K (regime 수) 찾기
        
        Metrics:
        - Silhouette Score (높을수록 좋음, -1~1)
        - Davies-Bouldin Index (낮을수록 좋음, 0~∞)
        - Calinski-Harabasz Index (높을수록 좋음, 0~∞)
        - Inertia (Elbow method)
        """
        print("\n" + "="*80)
        print("Optimal K (Regime 수) 탐색")
        print("="*80)
        
        if self.daily_summary is None:
            self.compute_market_summary()
        
        daily = self.daily_summary
        
        # Clustering features
        if features is None:
            features = [
                'market_return', 'market_volatility', 'vix',
                'treasury_10y', 'yield_spread',
                'ma_return_5d', 'ma_vol_5d',
                'ma_return_20d', 'ma_vol_20d'
            ]
        
        X = daily[features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Metrics 계산
        results = []
        
        print(f"\nK 범위: {list(k_range)}")
        print("\nClustering 진행 중...")
        
        for k in tqdm(k_range):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
            labels = kmeans.fit_predict(X_scaled)
            
            # Metrics
            silhouette = silhouette_score(X_scaled, labels)
            davies_bouldin = davies_bouldin_score(X_scaled, labels)
            calinski_harabasz = calinski_harabasz_score(X_scaled, labels)
            inertia = kmeans.inertia_
            
            # Regime 균형 (entropy)
            counts = pd.Series(labels).value_counts(normalize=True)
            entropy = -np.sum(counts * np.log(counts + 1e-10))
            max_entropy = np.log(k)
            balance = entropy / max_entropy  # 0~1, 1이 완전 균형
            
            results.append({
                'k': k,
                'silhouette': silhouette,
                'davies_bouldin': davies_bouldin,
                'calinski_harabasz': calinski_harabasz,
                'inertia': inertia,
                'balance': balance
            })
        
        results_df = pd.DataFrame(results)
        
        # 정규화 (0~1 스케일)
        results_df['silhouette_norm'] = (results_df['silhouette'] - results_df['silhouette'].min()) / \
                                         (results_df['silhouette'].max() - results_df['silhouette'].min())
        results_df['davies_bouldin_norm'] = 1 - (results_df['davies_bouldin'] - results_df['davies_bouldin'].min()) / \
                                                 (results_df['davies_bouldin'].max() - results_df['davies_bouldin'].min())
        results_df['calinski_harabasz_norm'] = (results_df['calinski_harabasz'] - results_df['calinski_harabasz'].min()) / \
                                                (results_df['calinski_harabasz'].max() - results_df['calinski_harabasz'].min())
        
        # Composite score (가중 평균)
        results_df['composite_score'] = (
            0.35 * results_df['silhouette_norm'] +
            0.25 * results_df['davies_bouldin_norm'] +
            0.25 * results_df['calinski_harabasz_norm'] +
            0.15 * results_df['balance']
        )
        
        # 결과 출력
        print("\n" + "="*80)
        print("K별 Clustering Metrics")
        print("="*80)
        print(results_df[['k', 'silhouette', 'davies_bouldin', 'calinski_harabasz', 
                          'balance', 'composite_score']].to_string(index=False))
        
        # 최적 K
        optimal_k = results_df.loc[results_df['composite_score'].idxmax(), 'k']
        print(f"\n✓ 최적 K (Composite Score 기준): {optimal_k}")
        
        # Elbow method 시각화
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Silhouette Score
        axes[0, 0].plot(results_df['k'], results_df['silhouette'], 'o-')
        axes[0, 0].set_xlabel('K (Number of Regimes)')
        axes[0, 0].set_ylabel('Silhouette Score')
        axes[0, 0].set_title('Silhouette Score (Higher is Better)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axvline(optimal_k, color='red', linestyle='--', alpha=0.5, label=f'Optimal K={optimal_k}')
        axes[0, 0].legend()
        
        # 2. Davies-Bouldin Index
        axes[0, 1].plot(results_df['k'], results_df['davies_bouldin'], 'o-')
        axes[0, 1].set_xlabel('K (Number of Regimes)')
        axes[0, 1].set_ylabel('Davies-Bouldin Index')
        axes[0, 1].set_title('Davies-Bouldin Index (Lower is Better)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axvline(optimal_k, color='red', linestyle='--', alpha=0.5, label=f'Optimal K={optimal_k}')
        axes[0, 1].legend()
        
        # 3. Calinski-Harabasz Index
        axes[0, 2].plot(results_df['k'], results_df['calinski_harabasz'], 'o-')
        axes[0, 2].set_xlabel('K (Number of Regimes)')
        axes[0, 2].set_ylabel('Calinski-Harabasz Index')
        axes[0, 2].set_title('Calinski-Harabasz Index (Higher is Better)')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].axvline(optimal_k, color='red', linestyle='--', alpha=0.5, label=f'Optimal K={optimal_k}')
        axes[0, 2].legend()
        
        # 4. Inertia (Elbow)
        axes[1, 0].plot(results_df['k'], results_df['inertia'], 'o-')
        axes[1, 0].set_xlabel('K (Number of Regimes)')
        axes[1, 0].set_ylabel('Inertia')
        axes[1, 0].set_title('Inertia (Elbow Method)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axvline(optimal_k, color='red', linestyle='--', alpha=0.5, label=f'Optimal K={optimal_k}')
        axes[1, 0].legend()
        
        # 5. Balance (Entropy)
        axes[1, 1].plot(results_df['k'], results_df['balance'], 'o-')
        axes[1, 1].set_xlabel('K (Number of Regimes)')
        axes[1, 1].set_ylabel('Balance (Normalized Entropy)')
        axes[1, 1].set_title('Regime Balance')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axvline(optimal_k, color='red', linestyle='--', alpha=0.5, label=f'Optimal K={optimal_k}')
        axes[1, 1].legend()
        
        # 6. Composite Score
        axes[1, 2].plot(results_df['k'], results_df['composite_score'], 'o-', linewidth=2)
        axes[1, 2].set_xlabel('K (Number of Regimes)')
        axes[1, 2].set_ylabel('Composite Score')
        axes[1, 2].set_title('Composite Score (Weighted Average)')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].axvline(optimal_k, color='red', linestyle='--', linewidth=2, label=f'Optimal K={optimal_k}')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig('./optimal_k_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ 시각화 저장: ./optimal_k_analysis.png")
        plt.close()
        
        # 결과 저장
        results_df.to_csv('./optimal_k_results.csv', index=False)
        print(f"✓ 결과 저장: ./optimal_k_results.csv")
        
        return results_df, optimal_k
    
    def analyze_purity_threshold(self, k=4, purity_range=[0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]):
        """
        Purity threshold 최적화
        
        Trade-off:
        - 높은 purity: 더 순수한 regime, 적은 episode 수
        - 낮은 purity: 더 많은 episode, regime 혼재
        """
        print("\n" + "="*80)
        print(f"Purity Threshold 분석 (K={k})")
        print("="*80)
        
        if self.daily_summary is None:
            self.compute_market_summary()
        
        # K-means로 regime 라벨링
        features = [
            'market_return', 'market_volatility', 'vix',
            'treasury_10y', 'yield_spread',
            'ma_return_5d', 'ma_vol_5d',
            'ma_return_20d', 'ma_vol_20d'
        ]
        
        X = self.daily_summary[features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        self.daily_summary['regime'] = kmeans.fit_predict(X_scaled)
        
        # Panel에 병합
        self.panel_data = self.panel_data.join(self.daily_summary[['regime']], how='left')
        
        # Episode 생성 시뮬레이션
        df = self.panel_data
        unique_dates = sorted(df.index.unique())
        
        support_len = 60
        query_len = 60
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
        
        results = []
        
        print(f"\nPurity threshold 범위: {purity_range}")
        print("\nEpisode 생성 시뮬레이션 중...")
        
        for purity in tqdm(purity_range):
            stats = {'purity_threshold': purity}
            
            for split_name, date_list in [('train', train_dates), 
                                          ('val', val_dates), 
                                          ('test', test_dates)]:
                n_valid = 0
                n_total_slots = len(date_list)
                regime_counts = {i: 0 for i in range(k)}
                
                for start_date in date_list:
                    start_idx_local = unique_dates.index(start_date) - support_len
                    support_dates = unique_dates[start_idx_local:start_idx_local + support_len]
                    query_start_idx = start_idx_local + support_len
                    query_dates = unique_dates[query_start_idx:query_start_idx + query_len]
                    
                    support_mask = df.index.isin(support_dates)
                    query_mask = df.index.isin(query_dates)
                    
                    support_regimes = df.loc[support_mask, 'regime'].dropna()
                    query_regimes = df.loc[query_mask, 'regime'].dropna()
                    
                    if len(support_regimes) == 0 or len(query_regimes) == 0:
                        continue
                    
                    sup_mode = support_regimes.mode()[0]
                    qry_mode = query_regimes.mode()[0]
                    
                    sup_purity = (support_regimes == sup_mode).mean()
                    qry_purity = (query_regimes == qry_mode).mean()
                    
                    # Regime-pure 조건
                    if sup_mode != qry_mode:
                        continue
                    if sup_purity < purity or qry_purity < purity:
                        continue
                    
                    n_valid += 1
                    regime_counts[int(sup_mode)] += 1
                
                stats[f'{split_name}_episodes'] = n_valid
                stats[f'{split_name}_ratio'] = n_valid / n_total_slots if n_total_slots > 0 else 0
                
                # Regime diversity (entropy)
                regime_dist = np.array([regime_counts[i] for i in range(k)])
                if regime_dist.sum() > 0:
                    regime_dist = regime_dist / regime_dist.sum()
                    entropy = -np.sum(regime_dist * np.log(regime_dist + 1e-10))
                    max_entropy = np.log(k)
                    diversity = entropy / max_entropy
                else:
                    diversity = 0
                
                stats[f'{split_name}_diversity'] = diversity
                stats[f'{split_name}_regime_dist'] = regime_counts
            
            results.append(stats)
        
        results_df = pd.DataFrame(results)
        
        # 출력
        print("\n" + "="*80)
        print("Purity Threshold별 Episode 통계")
        print("="*80)
        
        display_cols = ['purity_threshold', 
                       'train_episodes', 'train_ratio', 'train_diversity',
                       'val_episodes', 'val_ratio', 'val_diversity',
                       'test_episodes', 'test_ratio', 'test_diversity']
        print(results_df[display_cols].to_string(index=False))
        
        # 권장사항
        print("\n" + "="*80)
        print("권장사항")
        print("="*80)
        
        # Val/Test diversity가 가장 높은 purity
        val_test_diversity = results_df['val_diversity'] + results_df['test_diversity']
        best_diversity_idx = val_test_diversity.idxmax()
        best_diversity_purity = results_df.loc[best_diversity_idx, 'purity_threshold']
        
        print(f"\n1. Val/Test Diversity 최대화: purity = {best_diversity_purity}")
        print(f"   - Val diversity: {results_df.loc[best_diversity_idx, 'val_diversity']:.3f}")
        print(f"   - Test diversity: {results_df.loc[best_diversity_idx, 'test_diversity']:.3f}")
        
        # Episode 수가 충분한 범위에서 diversity 최대
        min_episodes = 100  # Train 최소 요구 episode 수
        sufficient = results_df[results_df['train_episodes'] >= min_episodes]
        if len(sufficient) > 0:
            best_balanced_idx = sufficient['val_diversity'].idxmax()
            best_balanced_purity = sufficient.loc[best_balanced_idx, 'purity_threshold']
            
            print(f"\n2. 균형잡힌 선택 (Train≥{min_episodes}): purity = {best_balanced_purity}")
            print(f"   - Train episodes: {sufficient.loc[best_balanced_idx, 'train_episodes']:.0f}")
            print(f"   - Val diversity: {sufficient.loc[best_balanced_idx, 'val_diversity']:.3f}")
        
        # 시각화
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Episode 수
        axes[0, 0].plot(results_df['purity_threshold'], results_df['train_episodes'], 'o-', label='Train')
        axes[0, 0].plot(results_df['purity_threshold'], results_df['val_episodes'], 's-', label='Val')
        axes[0, 0].plot(results_df['purity_threshold'], results_df['test_episodes'], '^-', label='Test')
        axes[0, 0].set_xlabel('Purity Threshold')
        axes[0, 0].set_ylabel('Number of Episodes')
        axes[0, 0].set_title('Episode Count vs Purity')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axvline(best_diversity_purity, color='red', linestyle='--', alpha=0.5)
        
        # 2. Episode ratio
        axes[0, 1].plot(results_df['purity_threshold'], results_df['train_ratio'], 'o-', label='Train')
        axes[0, 1].plot(results_df['purity_threshold'], results_df['val_ratio'], 's-', label='Val')
        axes[0, 1].plot(results_df['purity_threshold'], results_df['test_ratio'], '^-', label='Test')
        axes[0, 1].set_xlabel('Purity Threshold')
        axes[0, 1].set_ylabel('Valid Ratio')
        axes[0, 1].set_title('Valid Episode Ratio')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axvline(best_diversity_purity, color='red', linestyle='--', alpha=0.5)
        
        # 3. Diversity
        axes[1, 0].plot(results_df['purity_threshold'], results_df['train_diversity'], 'o-', label='Train')
        axes[1, 0].plot(results_df['purity_threshold'], results_df['val_diversity'], 's-', label='Val')
        axes[1, 0].plot(results_df['purity_threshold'], results_df['test_diversity'], '^-', label='Test')
        axes[1, 0].set_xlabel('Purity Threshold')
        axes[1, 0].set_ylabel('Regime Diversity')
        axes[1, 0].set_title('Regime Diversity (Normalized Entropy)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axvline(best_diversity_purity, color='red', linestyle='--', alpha=0.5, label=f'Best={best_diversity_purity}')
        
        # 4. Trade-off (Episode count vs Diversity for Val/Test)
        val_test_episodes = results_df['val_episodes'] + results_df['test_episodes']
        axes[1, 1].scatter(val_test_episodes, val_test_diversity, s=100, alpha=0.6)
        for idx, row in results_df.iterrows():
            axes[1, 1].annotate(f"{row['purity_threshold']:.2f}", 
                               (val_test_episodes[idx], val_test_diversity[idx]),
                               fontsize=9)
        axes[1, 1].set_xlabel('Val+Test Episodes')
        axes[1, 1].set_ylabel('Val+Test Diversity')
        axes[1, 1].set_title('Trade-off: Episode Count vs Diversity')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./purity_threshold_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ 시각화 저장: ./purity_threshold_analysis.png")
        plt.close()
        
        # 결과 저장
        results_df.to_csv('./purity_threshold_results.csv', index=False)
        print(f"✓ 결과 저장: ./purity_threshold_results.csv")
        
        return results_df, best_diversity_purity


# 실행
if __name__ == "__main__":
    print("전처리 최적화 분석")
    print("="*80)
    
    optimizer = PreprocessingOptimizer(panel_path='./sp500_data/sp500_panel.csv')
    
    # 1. 데이터 정제 및 Feature 선택
    print("\n" + "="*80)
    print("STEP 1: 데이터 정제 및 Feature 선택")
    print("="*80)
    df = optimizer.load_and_clean()
    
    # 2. Market summary 계산
    optimizer.compute_market_summary()
    
    # 3. Optimal K 탐색
    print("\n" + "="*80)
    print("STEP 2: Optimal K (Regime 수) 탐색")
    print("="*80)
    k_results, optimal_k = optimizer.find_optimal_k(k_range=range(2, 11))
    
    # 4. Purity threshold 분석
    print("\n" + "="*80)
    print("STEP 3: Purity Threshold 최적화")
    print("="*80)
    
    # Optimal K와 K=5 모두 분석
    for k in [5, int(optimal_k)]:
        print(f"\n{'='*80}")
        print(f"K={k} 분석")
        print('='*80)
        purity_results, best_purity = optimizer.analyze_purity_threshold(
            k=k, 
            purity_range=[0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
        )
    
    print("\n" + "="*80)
    print("분석 완료!")
    print("="*80)
    print("\n생성된 파일:")
    print("  📊 optimal_k_analysis.png")
    print("  📊 optimal_k_results.csv")
    print("  📊 purity_threshold_analysis.png")
    print("  📊 purity_threshold_results.csv")
    
    print("\n최종 권장사항:")
    print(f"  ✓ K (regime 수): {optimal_k}")
    print(f"  ✓ Purity threshold: {best_purity}")
    print(f"  ✓ Features: {len([col for col in df.columns if col not in ['ticker', 'returns']])}")
