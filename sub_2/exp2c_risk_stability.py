import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
import os

def get_hrp_weights(cov, corr):
    """
    Hierarchical Risk Parity 알고리즘
    """
    # 1. Clustering
    dist = np.sqrt((1 - corr) / 2)
    dist = np.nan_to_num(dist)
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0)
    link = linkage(squareform(dist, checks=False), method='ward')
    sort_ix = leaves_list(link)
    
    # 2. Recursive Bisection
    weights = pd.Series(1.0, index=sort_ix) # float 초기화
    
    def get_cluster_var(cov, c_items):
        cov_slice = cov[c_items][:, c_items]
        # [수정] Singular Matrix 방지를 위해 pinv 사용
        w = np.linalg.pinv(cov_slice).sum(axis=1)
        w /= w.sum() # IVP weights
        return np.dot(np.dot(w.T, cov_slice), w)
        
    def recurse_bi_section(w, c_items):
        if len(c_items) == 1:
            return
        split = len(c_items) // 2
        left = c_items[:split]
        right = c_items[split:]
        
        left_var = get_cluster_var(cov, left)
        right_var = get_cluster_var(cov, right)
        
        # 분모가 0이 되는 경우 방지
        if left_var + right_var == 0:
            alpha = 0.5
        else:
            alpha = 1 - left_var / (left_var + right_var)
            
        w[left] *= alpha
        w[right] *= (1 - alpha)
        
        recurse_bi_section(w, left)
        recurse_bi_section(w, right)
        
    recurse_bi_section(weights, sort_ix)
    return weights.sort_index().values

def run_risk_stability_experiment():
    print("="*80)
    print("Experiment 2C: Out-of-Sample Risk Stability (HRP vs MVO)")
    print("="*80)
    
    # 1. 데이터 로드
    panel_path = './processed_data/K4_tau0.60/sp500_panel_K4.csv'
    if not os.path.exists(panel_path):
        print("Data file not found.")
        return

    df = pd.read_csv(panel_path)
    returns = df.pivot(index='date', columns='ticker', values='returns').fillna(0)
    
    # Train/Test Split (마지막 20%를 Test로)
    n_train = int(len(returns) * 0.8)
    test_returns = returns.iloc[n_train:]
    
    print(f"Test Period: {len(test_returns)} days")
    
    # 2. Rolling Backtest
    window = 252
    rebalance_freq = 20
    
    strategies = {
        'Equal Weight': [],
        'Sample MVO (MinVar)': [],
        'HRP': []
    }
    
    dates = []
    
    print("Running Rolling Backtest...")
    for i in range(window, len(test_returns), rebalance_freq):
        # Lookback Window
        train_slice = test_returns.iloc[i-window:i]
        next_slice = test_returns.iloc[i:i+rebalance_freq]
        
        if len(next_slice) == 0: break
        
        # Covariance & Correlation
        cov = train_slice.cov().values
        corr = train_slice.corr().values
        
        # 1. Equal Weight
        w_ew = np.ones(len(cov)) / len(cov)
        
        # 2. Sample MVO (Global Min Variance)
        try:
            inv_cov = np.linalg.pinv(cov) # 여기도 pinv
            w_mvo = inv_cov.sum(axis=1)
            # 음수 비중 제거 (Long-only constraint) -> 간단히 0으로 자르고 정규화
            w_mvo[w_mvo < 0] = 0
            w_mvo /= w_mvo.sum()
        except:
            w_mvo = w_ew 
            
        # 3. HRP
        try:
            w_hrp = get_hrp_weights(cov, corr)
        except Exception as e:
            print(f"HRP Error at {i}: {e}")
            w_hrp = w_ew
        
        # Calculate Realized Volatility (Annualized)
        for name, w in [('Equal Weight', w_ew), ('Sample MVO (MinVar)', w_mvo), ('HRP', w_hrp)]:
            # Portfolio Returns
            port_ret = next_slice.dot(w)
            vol = port_ret.std() * np.sqrt(252)
            strategies[name].append(vol)
            
        dates.append(test_returns.index[i])
        
    # 3. 결과 분석
    results = pd.DataFrame(strategies, index=dates)
    
    print("\n[Average Annualized Volatility]")
    print(results.mean())
    
    # 시각화
    plt.figure(figsize=(10, 6))
    results.mean().plot(kind='bar', color=['gray', 'red', 'blue'], alpha=0.7)
    plt.title('Out-of-Sample Portfolio Volatility Comparison')
    plt.ylabel('Annualized Volatility')
    plt.grid(True, alpha=0.3, axis='y')
    
    os.makedirs('./results', exist_ok=True)
    plt.savefig('./results/exp2c_risk_stability.png')
    print("\n✓ Plot saved to ./results/exp2c_risk_stability.png")
    
    hrp_vol = results['HRP'].mean()
    mvo_vol = results['Sample MVO (MinVar)'].mean()
    
    improvement = (mvo_vol - hrp_vol) / mvo_vol * 100
    print(f"\nImprovement: {improvement:.2f}% reduction in volatility vs MVO")
    
    if hrp_vol < mvo_vol:
        print(">> RESULT: HRP reduces risk compared to MVO. Structural stability confirmed.")
    else:
        print(">> RESULT: HRP did not outperform MVO.")

if __name__ == "__main__":
    run_risk_stability_experiment()