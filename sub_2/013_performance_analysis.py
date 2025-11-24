import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def calculate_metrics(returns, turnover=None):
    """
    금융 성과 지표 계산
    returns: 일별 수익률 Series
    turnover: 일별 회전율 Series (Optional)
    """
    # 1. Annualized Return
    # 복리 수익률 (CAGR)
    cum_ret = (1 + returns).prod()
    n_years = len(returns) / 252
    ann_ret = cum_ret ** (1 / n_years) - 1
    
    # 2. Annualized Volatility
    ann_vol = returns.std() * np.sqrt(252)
    
    # 3. Sharpe Ratio (Rf=0 가정)
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    
    # 4. Sortino Ratio
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino = (ann_ret / downside_std) if downside_std > 0 else np.nan
    
    # 5. Max Drawdown
    cum_returns = (1 + returns).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    mdd = drawdown.min()
    
    # 6. Win Rate
    win_rate = (returns > 0).mean()
    
    # 7. Turnover (Annualized)
    ann_turnover = turnover.mean() * 252 if turnover is not None else 0
    
    return {
        'CAGR': ann_ret,
        'Volatility': ann_vol,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'MDD': mdd,
        'WinRate': win_rate,
        'Turnover': ann_turnover
    }

def run_performance_analysis():
    print("="*80)
    print("Performance Analysis & Visualization")
    print("="*80)
    
    # 1. 결과 로드
    result_path = './results/backtest_results_strict.csv'
    if not os.path.exists(result_path):
        print("Result file not found.")
        return
        
    df = pd.read_csv(result_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # 전략 목록 추출 (컬럼명: {Strategy}_Return)
    strategies = [col.replace('_Return', '') for col in df.columns if '_Return' in col]
    
    metrics_list = []
    
    # 2. 지표 계산
    for strat in strategies:
        ret_col = f'{strat}_Return'
        to_col = f'{strat}_Turnover'
        
        metrics = calculate_metrics(df[ret_col], df[to_col])
        metrics['Strategy'] = strat
        metrics_list.append(metrics)
        
    metrics_df = pd.DataFrame(metrics_list).set_index('Strategy')
    
    # 출력
    print("\n[Performance Metrics]")
    print(metrics_df.round(4))
    
    # CSV 저장
    metrics_df.to_csv('./results/performance_metrics.csv')
    
    # 3. 시각화
    # (1) Cumulative Returns (Log Scale Optional)
    plt.figure(figsize=(12, 6))
    for strat in strategies:
        cum_ret = (1 + df[f'{strat}_Return']).cumprod()
        plt.plot(cum_ret, label=strat, linewidth=2)
        
    plt.title('Cumulative Returns (Out-of-Sample)')
    plt.xlabel('Date')
    plt.ylabel('Wealth Index (Start=1.0)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('./results/cumulative_returns.png')
    print("✓ Saved cumulative_returns.png")
    
    # (2) Drawdown Plot
    plt.figure(figsize=(12, 4))
    for strat in strategies:
        cum_ret = (1 + df[f'{strat}_Return']).cumprod()
        peak = cum_ret.cummax()
        dd = (cum_ret - peak) / peak
        plt.plot(dd, label=strat, linewidth=1)
        
    plt.title('Drawdown History')
    plt.ylabel('Drawdown')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.fill_between(df.index, 0, -1, color='gray', alpha=0.1)
    plt.savefig('./results/drawdown.png')
    print("✓ Saved drawdown.png")
    
    # (3) Rolling Sharpe (6-month)
    plt.figure(figsize=(12, 4))
    window = 126 # 6 months
    for strat in strategies:
        roll_mean = df[f'{strat}_Return'].rolling(window).mean()
        roll_std = df[f'{strat}_Return'].rolling(window).std()
        roll_sharpe = (roll_mean / roll_std) * np.sqrt(252)
        plt.plot(roll_sharpe, label=strat, linewidth=1.5)
        
    plt.title(f'Rolling Sharpe Ratio ({window}-day window)')
    plt.ylabel('Sharpe Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('./results/rolling_sharpe.png')
    print("✓ Saved rolling_sharpe.png")

if __name__ == "__main__":
    run_performance_analysis()