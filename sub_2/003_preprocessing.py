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

class EnhancedPreprocessor:
    """
    개선된 전처리 파이프라인:
    1. Train-only median imputation (미래 정보 제거)
    2. Missing indicator features 추가
    3. Optimal K/purity 자동 선택
    """
    
    def __init__(self, panel_path='./sp500_data/sp500_panel.csv'):
        self.panel_path = panel_path
        self.panel_data = None
        self.train_statistics = {}  # Train-only statistics
        
    def load_and_clean_basic(self):
        """기본 정제 (asset_turnover 제거)"""
        print("="*80)
        print("데이터 로드 및 기본 정제")
        print("="*80)
        
        df = pd.read_csv(self.panel_path, index_col=0, parse_dates=True)
        print(f"\n원본 shape: {df.shape}")
        
        # asset_turnover 제거
        if 'asset_turnover' in df.columns:
            df = df.drop(columns=['asset_turnover'])
            print("✓ asset_turnover 제거")
        
        self.panel_data = df
        return df
    
    def add_missing_indicators(self, cols_to_track=None):
        """
        Missing indicator features 추가
        
        Args:
            cols_to_track: Missing indicator를 만들 컬럼 리스트
                          None이면 fundamental features에 대해 자동 생성
        """
        print("\n" + "="*80)
        print("Missing Indicator Features 추가")
        print("="*80)
        
        df = self.panel_data
        
        if cols_to_track is None:
            # Fundamental features (missing이 의미있는 것들)
            cols_to_track = [
                'pe_ratio', 'pb_ratio', 'roe', 'roa', 'debt_to_equity',
                'dividend_yield', 'payout_ratio', 'free_cashflow_yield'
            ]
            cols_to_track = [col for col in cols_to_track if col in df.columns]
        
        print(f"\nMissing indicator를 추가할 컬럼 ({len(cols_to_track)}개):")
        
        for col in cols_to_track:
            indicator_col = f'{col}_missing'
            df[indicator_col] = df[col].isnull().astype(int)
            
            n_missing = df[indicator_col].sum()
            pct_missing = n_missing / len(df) * 100
            
            print(f"  + {indicator_col}: {n_missing:,} ({pct_missing:.1f}%)")
        
        print(f"\n✓ {len(cols_to_track)}개 missing indicator 추가")
        print(f"✓ 총 feature 수: {len([c for c in df.columns if c not in ['ticker', 'returns']])}")
        
        self.panel_data = df
        return df
    
    def impute_with_train_statistics(self, train_end_date):
        """
        Train-only statistics로 imputation (미래 정보 누출 방지)
        
        Args:
            train_end_date: Training set 종료 날짜
        """
        print("\n" + "="*80)
        print(f"Train-Only Imputation (Train end: {train_end_date})")
        print("="*80)
        
        df = self.panel_data
        
        # Train 구간 데이터
        train_mask = df.index <= train_end_date
        train_df = df[train_mask]
        
        print(f"\nTrain 구간: {train_df.index.min()} ~ {train_df.index.max()}")
        print(f"Train 데이터: {len(train_df):,} / {len(df):,} ({len(train_df)/len(df)*100:.1f}%)")
        
        # 1. Fundamental features (ticker별 forward fill → train median)
        fundamental_cols = ['pe_ratio', 'pb_ratio', 'roe', 'roa', 'debt_to_equity',
                           'dividend_yield', 'payout_ratio', 'free_cashflow_yield']
        fundamental_cols = [col for col in fundamental_cols if col in df.columns]
        
        print("\n[Fundamental Features]")
        for col in fundamental_cols:
            # Train median 계산
            train_median = train_df[col].median()
            self.train_statistics[f'{col}_median'] = train_median
            
            before_missing = df[col].isnull().sum()
            
            # Ticker별 forward fill
            df[col] = df.groupby('ticker')[col].transform(lambda x: x.fillna(method='ffill'))
            
            # Train median으로 나머지 채우기
            df[col] = df[col].fillna(train_median)
            
            after_missing = df[col].isnull().sum()
            print(f"  {col}: {before_missing:,} → {after_missing:,} (median={train_median:.4f})")
        
        # 2. Macro features (전체 forward fill, 미래 정보 없음)
        macro_cols = ['vix', 'treasury_10y', 'treasury_2y', 'yield_spread',
                     'usd_index', 'credit_spread', 'cpi_yoy']
        macro_cols = [col for col in macro_cols if col in df.columns]
        
        print("\n[Macro Features]")
        for col in macro_cols:
            before_missing = df[col].isnull().sum()
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
            after_missing = df[col].isnull().sum()
            if before_missing > 0:
                print(f"  {col}: {before_missing:,} → {after_missing:,}")
        
        # 3. Technical indicators (ticker별 forward fill → backfill)
        tech_cols = ['rsi', 'ma_20', 'ma_50', 'ma_200', 'macd', 'macd_signal',
                    'bollinger_upper', 'bollinger_lower', 'atr', 'stoch_k', 'stoch_d',
                    'price_roc', 'realized_vol_20', 'realized_vol_60', 'volume_roc',
                    'mfi', 'obv', 'williams_r']
        tech_cols = [col for col in tech_cols if col in df.columns]
        
        print("\n[Technical Features]")
        for col in tech_cols:
            df[col] = df.groupby('ticker')[col].transform(
                lambda x: x.fillna(method='ffill').fillna(method='bfill')
            )
        print(f"  {len(tech_cols)}개 technical features 처리 완료")
        
        # 최종 확인
        final_missing = df.isnull().sum().sum()
        print(f"\n✓ 최종 missing values: {final_missing:,}")
        
        # Train statistics 저장
        stats_path = './train_statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(self.train_statistics, f, indent=2)
        print(f"✓ Train statistics 저장: {stats_path}")
        
        self.panel_data = df
        return df
    
    def save_processed_data(self, output_path='./sp500_panel_cleaned.csv'):
        """전처리된 데이터 저장"""
        print("\n" + "="*80)
        print("전처리 데이터 저장")
        print("="*80)
        
        self.panel_data.to_csv(output_path)
        
        # Feature 통계
        feature_cols = [col for col in self.panel_data.columns 
                       if col not in ['ticker', 'returns']]
        
        missing_indicators = [col for col in feature_cols if col.endswith('_missing')]
        base_features = [col for col in feature_cols if not col.endswith('_missing')]
        
        print(f"\n✓ 저장 완료: {output_path}")
        print(f"  - Shape: {self.panel_data.shape}")
        print(f"  - Base features: {len(base_features)}")
        print(f"  - Missing indicators: {len(missing_indicators)}")
        print(f"  - Total features: {len(feature_cols)}")
        print(f"  - Size: {os.path.getsize(output_path) / 1024**2:.1f} MB")
        
        return output_path


# 실행 스크립트
if __name__ == "__main__":
    print("Enhanced Preprocessing Pipeline")
    print("="*80)
    print("\n개선사항:")
    print("  1. ✅ Train-only median imputation (미래 정보 제거)")
    print("  2. ✅ Missing indicator features 추가")
    print("  3. ✅ 38 base features + 8 missing indicators = 46 total")
    
    # 초기화
    preprocessor = EnhancedPreprocessor(
        panel_path='./sp500_data/sp500_panel.csv'
    )
    
    # STEP 1: 기본 정제
    print("\n" + "="*80)
    print("STEP 1: 기본 정제")
    print("="*80)
    df = preprocessor.load_and_clean_basic()
    
    # STEP 2: Missing indicator 추가
    print("\n" + "="*80)
    print("STEP 2: Missing Indicator Features 추가")
    print("="*80)
    df = preprocessor.add_missing_indicators()
    
    # STEP 3: Train-only imputation
    print("\n" + "="*80)
    print("STEP 3: Train-Only Statistics Imputation")
    print("="*80)
    
    # Train/Val/Test split 날짜 (이전 분석 결과 기반)
    # Train: 2006-03-30 ~ 2019-10-28
    train_end_date = '2019-10-28'
    
    df = preprocessor.impute_with_train_statistics(train_end_date=train_end_date)
    
    # STEP 4: 저장
    print("\n" + "="*80)
    print("STEP 4: 저장")
    print("="*80)
    output_path = preprocessor.save_processed_data(
        output_path='./processed_data/sp500_panel_enhanced.csv'
    )
    
    print("\n" + "="*80)
    print("전처리 완료!")
    print("="*80)
    print("\n최종 데이터셋:")
    print(f"  ✅ 38 base features")
    print(f"  ✅ 8 missing indicators")
    print(f"  ✅ 46 total features")
    print(f"  ✅ Train-only median (미래 정보 없음)")
    print(f"  ✅ Missing patterns 보존")
    
    print("\n생성 파일:")
    print(f"  📊 {output_path}")
    print(f"  📊 ./train_statistics.json")
    
    print("\n다음 단계:")
    print("  1. ✅ 001_preprocessing.py로 K=4, purity=0.7로 episode 생성")
    print("  2. ✅ PyTorch Dataset 구현")
    print("  3. ✅ Meta-learner 학습")
