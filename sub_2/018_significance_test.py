import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp

def bootstrap_sharpe_difference(ret_A, ret_B, n_bootstrap=10000, block_size=10):
    """
    Ledoit-Wolf style Moving Block Bootstrap for Sharpe Ratio Difference
    H0: Sharpe(A) = Sharpe(B)
    """
    n = len(ret_A)
    diff_original = (np.mean(ret_A) / np.std(ret_A)) - (np.mean(ret_B) / np.std(ret_B))
    
    # Combined returns for centering
    # (Bootstrap에서는 귀무가설 하에서의 분포를 만들어야 하므로, 평균을 조정하기도 하지만
    # 여기서는 단순 차이의 신뢰구간을 구해서 0이 포함되는지 보는 방식 사용)
    
    diffs = []
    indices = np.arange(n)
    
    for _ in range(n_bootstrap):
        # Moving Block Bootstrap
        # 블록 시작점 랜덤 선택
        starts = np.random.randint(0, n - block_size + 1, int(n / block_size) + 1)
        boot_idx = []
        for s in starts:
            boot_idx.extend(indices[s : s + block_size])
        boot_idx = boot_idx[:n]
        
        rA_b = ret_A[boot_idx]
        rB_b = ret_B[boot_idx]
        
        # Sharpe Difference in this bootstrap sample
        # (연율화 상수는 약분되므로 생략 가능하나 명시적으로 포함)
        sharpe_A = (np.mean(rA_b) / np.std(rA_b)) * np.sqrt(252)
        sharpe_B = (np.mean(rB_b) / np.std(rB_b)) * np.sqrt(252)
        
        diffs.append(sharpe_A - sharpe_B)
        
    diffs = np.array(diffs)
    
    # p-value (Two-sided)
    # 0을 중심으로 얼마나 치우쳐 있는지
    # H0: diff = 0
    # p = 2 * min(P(diff > 0), P(diff < 0))
    p_val = 2 * min(np.mean(diffs > 0), np.mean(diffs < 0))
    
    # Confidence Interval
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)
    
    return {
        'diff_original': diff_original * np.sqrt(252),
        'p_value': p_val,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }

def run_significance_test():
    print("="*80)
    print("Statistical Significance Test (Bootstrap Sharpe Difference)")
    print("="*80)
    
    # 결과 로드
    df = pd.read_csv('./results/backtest_results_strict.csv')
    
    # Meta vs Pooled
    res_pooled = bootstrap_sharpe_difference(df['Meta_Return'].values, df['Pooled_Return'].values)
    
    # Meta vs EW
    res_ew = bootstrap_sharpe_difference(df['Meta_Return'].values, df['EW_Return'].values)
    
    print("\n[Test 1: Meta vs Pooled]")
    print(f"Sharpe Diff: {res_pooled['diff_original']:.4f}")
    print(f"95% CI: [{res_pooled['ci_lower']:.4f}, {res_pooled['ci_upper']:.4f}]")
    print(f"P-Value: {res_pooled['p_value']:.4f}")
    if res_pooled['p_value'] < 0.05:
        print(">> RESULT: Statistically Significant (p < 0.05) ***")
    else:
        print(">> RESULT: Not Significant")
        
    print("\n[Test 2: Meta vs Equal Weight]")
    print(f"Sharpe Diff: {res_ew['diff_original']:.4f}")
    print(f"95% CI: [{res_ew['ci_lower']:.4f}, {res_ew['ci_upper']:.4f}]")
    print(f"P-Value: {res_ew['p_value']:.4f}")
    if res_ew['p_value'] < 0.05:
        print(">> RESULT: Statistically Significant (p < 0.05) ***")
    else:
        print(">> RESULT: Not Significant")

if __name__ == "__main__":
    run_significance_test()