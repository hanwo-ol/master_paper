import pandas as pd
import numpy as np
import os

def check_data_integrity():
    print("="*80)
    print("Data Integrity Check")
    print("="*80)
    
    file_path = './processed_data/K4_tau0.60/sp500_panel_K4.csv'
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path)
    
    # 1. NaN / Inf Check
    print("\n[1] NaN / Inf Check")
    null_counts = df.isnull().sum()
    inf_counts = np.isinf(df.select_dtypes(include=np.number)).sum()
    
    total_nulls = null_counts.sum()
    total_infs = inf_counts.sum()
    
    print(f"Total NaNs: {total_nulls}")
    print(f"Total Infs: {total_infs}")
    
    if total_nulls > 0:
        print("\nColumns with NaNs:")
        print(null_counts[null_counts > 0])
        
    if total_infs > 0:
        print("\nColumns with Infs:")
        print(inf_counts[inf_counts > 0])

    # 2. Feature Statistics (Extreme Values)
    print("\n[2] Feature Statistics (Top 5 Max Values)")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats = df[numeric_cols].describe().T[['min', 'max', 'mean', 'std']]
    
    # Max 값이 1000을 넘는 컬럼 확인 (StandardScaler를 썼다면 대부분 작아야 함)
    high_val_cols = stats[stats['max'] > 100]
    if not high_val_cols.empty:
        print("\nWarning: Columns with very high values (> 100):")
        print(high_val_cols)
    else:
        print("\n✓ All features seem to be within reasonable range.")
        
    # 3. Returns Check
    print("\n[3] Returns Distribution")
    print(df['returns'].describe())
    
    # 극단적인 수익률 확인 (예: 100% 이상 상승/하락)
    extreme_returns = df[abs(df['returns']) > 1.0]
    if not extreme_returns.empty:
        print(f"\nWarning: Found {len(extreme_returns)} extreme returns (> 100%)")
        print(extreme_returns[['date', 'ticker', 'returns']].head())

if __name__ == "__main__":
    check_data_integrity()