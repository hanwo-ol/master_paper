import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

def run_covariance_stability_experiment():
    print("="*80)
    print("Experiment 2: HRP vs Random Ordering Stability Analysis")
    print("="*80)
    
    # 1. 데이터 로드 (Panel Data)
    panel_path = './processed_data/K4_tau0.60/sp500_panel_K4.csv'
    if not os.path.exists(panel_path):
        print(f"Error: File not found at {panel_path}")
        return

    df = pd.read_csv(panel_path)
    
    # Pivot to (Date x Ticker) returns
    returns_df = df.pivot(index='date', columns='ticker', values='returns')
    returns_df = returns_df.fillna(0)
    
    # 전체 공분산 행렬 계산
    print("Computing full covariance matrix...")
    full_cov = returns_df.cov().values
    n_assets = full_cov.shape[0]
    
    # 2. HRP Ordering 계산
    print("Computing HRP Ordering...")
    
    # Correlation -> Distance
    corr = returns_df.corr().values
    dist = np.sqrt((1 - corr) / 2)
    dist = np.nan_to_num(dist)
    
    # Hierarchical Clustering
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0)
    
    condensed_dist = squareform(dist, checks=False)
    link = linkage(condensed_dist, method='ward')
    hrp_order = leaves_list(link)
    
    # Random Ordering
    random_order = np.random.permutation(n_assets)
    
    # 3. Local Condition Number 계산 함수
    def get_local_condition_numbers(cov_matrix, ordering, window_size=20):
        ordered_cov = cov_matrix[ordering][:, ordering]
        cond_nums = []
        
        # 대각선을 따라 윈도우 슬라이딩
        for i in range(0, n_assets - window_size + 1, 5): # Stride 5
            sub_cov = ordered_cov[i:i+window_size, i:i+window_size]
            
            # Condition Number = lambda_max / lambda_min
            evals = np.linalg.eigvalsh(sub_cov)
            evals = evals[evals > 1e-8]
            
            if len(evals) > 0:
                cond = evals.max() / evals.min()
                cond_nums.append(cond)
                
        return cond_nums
    
    print("Calculating Local Condition Numbers...")
    window_size = 20 
    
    hrp_conds = get_local_condition_numbers(full_cov, hrp_order, window_size)
    rand_conds = get_local_condition_numbers(full_cov, random_order, window_size)
    
    # 4. 결과 시각화
    plt.figure(figsize=(10, 6))
    sns.kdeplot(np.log10(hrp_conds), fill=True, label='HRP Ordering', color='blue')
    sns.kdeplot(np.log10(rand_conds), fill=True, label='Random Ordering', color='red')
    
    plt.title(f'Distribution of Local Condition Numbers (Window Size={window_size})')
    plt.xlabel('Log10(Condition Number)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs('./results', exist_ok=True)
    plt.savefig('./results/exp2_covariance_stability.png')
    print("\n✓ Plot saved to ./results/exp2_covariance_stability.png")
    
    # 통계 출력
    print(f"\n[Statistics - Log10 Condition Number]")
    print(f"HRP    Mean: {np.mean(np.log10(hrp_conds)):.4f}, Std: {np.std(np.log10(hrp_conds)):.4f}")
    print(f"Random Mean: {np.mean(np.log10(rand_conds)):.4f}, Std: {np.std(np.log10(rand_conds)):.4f}")
    
    improvement = (np.mean(np.log10(rand_conds)) - np.mean(np.log10(hrp_conds))) / np.mean(np.log10(rand_conds)) * 100
    print(f"\n>> Improvement in Conditioning: {improvement:.2f}% reduction in log-condition number.")
    
    if improvement > 0:
        print(">> RESULT: HRP Ordering improves local conditioning. Proposition 2 supported.")
    else:
        print(">> RESULT: No significant improvement observed.")

if __name__ == "__main__":
    run_covariance_stability_experiment()