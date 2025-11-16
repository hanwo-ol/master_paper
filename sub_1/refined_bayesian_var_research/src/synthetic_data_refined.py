# ============================================================================
# synthetic_data_refined.py
# Stage 2: Synthetic Data Generation with Extreme Value Analysis
# ============================================================================

import numpy as np
import pandas as pd
from typing import Dict, Tuple


class VaRLabelGenerator:
    """VaR 레이블 생성"""
    
    def __init__(self, rolling_window: int = 252):
        self.rolling_window = rolling_window
    
    def generate_labels(self, portfolio_returns: Dict[str, pd.Series]) -> Dict[str, np.ndarray]:
        """Historical VaR 계산"""
        labels = {}
        
        for portfolio_name, returns in portfolio_returns.items():
            var_95 = returns.rolling(window=self.rolling_window).quantile(0.05)
            labels[portfolio_name] = var_95.dropna().values
        
        return labels
    
    def print_label_statistics(self, labels: Dict) -> None:
        """레이블 통계 출력"""
        print("\n【VaR Label Statistics】")
        print(f"{'Portfolio':<20} {'Mean VaR':<15} {'Std VaR':<15} {'Min VaR':<15}")
        print("-" * 65)
        
        for portfolio_name, var_values in labels.items():
            print(f"{portfolio_name:<20} {var_values.mean():>14.4%}  {var_values.std():>14.4%}  {var_values.min():>14.4%}")


class SyntheticDataGenerator:
    """합성 데이터 생성"""
    
    def __init__(self, returns: pd.DataFrame, n_scenarios: int = 100000):
        self.returns = returns
        self.n_scenarios = n_scenarios
        self.mean = returns.mean()
        self.cov = returns.cov()
        self.std = returns.std()
    
    def generate_scenarios(self) -> np.ndarray:
        """Multivariate normal에서 합성 포트폴리오 시나리오 생성"""
        scenarios = np.random.multivariate_normal(
            self.mean, self.cov, size=self.n_scenarios
        )
        return scenarios
    
    def extract_features(self, scenarios: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """포트폴리오 특성 추출 (11D)"""
        
        n_features = 11
        features = np.zeros((len(scenarios), n_features))
        
        # Features:
        # [0-7]: 8개 자산 가중치
        # [8]: 평균 일일 수익
        # [9]: 변동성
        # [10]: 공분산 합
        
        features[:, :8] = weights  # 가중치
        
        for i, scenario in enumerate(scenarios):
            portfolio_returns = np.dot(scenario, weights)
            features[i, 8] = np.mean(scenario)  # 평균 수익
            features[i, 9] = np.std(scenario)   # 변동성
            features[i, 10] = np.sum(self.cov.values) / (len(self.cov) ** 2)  # 공분산 합
        
        return features


def create_training_dataset(train_returns: pd.DataFrame, val_ratio: float = 0.2) -> Dict:
    """
    훈련용 데이터셋 생성
    
    Processes:
    1. 5가지 포트폴리오 타입 정의
    2. 각 포트폴리오 타입별 특성 추출
    3. 합성 데이터 생성 (100K scenarios)
    4. Train/Val 분할
    5. 정규화
    """
    
    print("\n【Creating Synthetic Training Dataset】")
    
    # 포트폴리오 타입
    portfolio_types = {
        'Balanced': np.array([0.15, 0.15, 0.15, 0.15, 0.15, 0.1, 0.1, 0.05]),
        'Aggressive': np.array([0.2, 0.2, 0.1, 0.05, 0.25, 0.1, 0.05, 0.05]),
        'Conservative': np.array([0.1, 0.1, 0.2, 0.2, 0.0, 0.05, 0.2, 0.15]),
        'Tech-Heavy': np.array([0.25, 0.25, 0.0, 0.0, 0.3, 0.15, 0.0, 0.05]),
        'Safe-Haven': np.array([0.05, 0.05, 0.1, 0.1, 0.0, 0.0, 0.3, 0.4])
    }
    
    # 합성 데이터 생성
    generator = SyntheticDataGenerator(train_returns, n_scenarios=100000)
    scenarios = generator.generate_scenarios()
    
    print(f"[OK] Generated {len(scenarios):,} synthetic scenarios")
    
    # 모든 포트폴리오별 특성 추출
    all_features = []
    all_labels = []
    
    for portfolio_name, weights in portfolio_types.items():
        # VaR 계산 (각 시나리오의 포트폴리오 수익률)
        portfolio_returns_sim = np.dot(scenarios, weights)
        
        # VaR 95% (5% 분위수 = 일일 손실)
        var_labels = np.percentile(portfolio_returns_sim, 5, axis=0) if portfolio_returns_sim.ndim > 1 \
                     else np.percentile(portfolio_returns_sim, 5)
        
        # 특성 추출
        features = generator.extract_features(scenarios, weights)
        
        all_features.append(features)
        all_labels.extend([var_labels] * len(features))
    
    # 합치기
    X = np.vstack(all_features)
    y = np.array(all_labels)
    
    # 정규화
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_norm = (X - X_mean) / (X_std + 1e-8)
    
    # Train/Val 분할
    n_train = int(len(X_norm) * (1 - val_ratio))
    indices = np.random.permutation(len(X_norm))
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    dataset = {
        'X_train': X_norm[train_idx].astype(np.float32),
        'y_train': y[train_idx].astype(np.float32),
        'X_val': X_norm[val_idx].astype(np.float32),
        'y_val': y[val_idx].astype(np.float32),
        'X_mean': X_mean,
        'X_std': X_std
    }
    
    print(f"[OK] Dataset created:")
    print(f"     Training: {len(dataset['X_train']):,} samples")
    print(f"     Validation: {len(dataset['X_val']):,} samples")
    print(f"     Feature dimension: {dataset['X_train'].shape[1]}D")
    
    return dataset


def main():
    """Main execution"""
    print("Loading market returns...")
    import yfinance as yf
    
    # 데이터 로드 (간단한 예시)
    tickers = ['AAPL', 'MSFT', 'JPM', 'PG', 'TSLA', 'AMD', 'GLD', 'TLT']
    data = yf.download(tickers, start='2019-01-01', end='2025-11-16', progress=False)
    prices = data['Adj Close']
    returns = prices.pct_change().dropna()
    
    # 데이터셋 생성
    dataset = create_training_dataset(returns, val_ratio=0.2)
    
    return dataset


if __name__ == '__main__':
    dataset = main()
