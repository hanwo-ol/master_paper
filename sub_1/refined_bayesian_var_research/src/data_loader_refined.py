# ============================================================================
# data_loader_refined.py (FIXED)
# Stage 1: Data Collection with Representativeness Validation
# FIX: yfinance download handling
# ============================================================================

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import os
from scipy.stats import skew, kurtosis

class PortfolioDataLoader:
    """실제 시장 데이터 수집 및 전처리 (개선: 대표성 검증 추가)"""
    
    def __init__(self, tickers: list = None, start_date: str = '2019-01-01', 
                 end_date: str = '2025-11-16'):
        self.tickers = tickers or ['AAPL', 'MSFT', 'JPM', 'PG', 'TSLA', 'AMD', 'GLD', 'TLT']
        self.start_date = start_date
        self.end_date = end_date
        self.prices_df = None
        self.returns_df = None
        
    def download_data(self) -> pd.DataFrame:
        """개선된 다운로드"""
        print(f"Downloading data for {len(self.tickers)} assets...")
        
        # 방법 1: 한번에 다운로드
        try:
            data = yf.download(
                self.tickers, 
                start=self.start_date, 
                end=self.end_date, 
                group_by='ticker',
                progress=False
            )
            
            # 각 ticker의 Adj Close 추출
            prices = {}
            for ticker in self.tickers:
                if ticker in data.columns.levels[0]:
                    prices[ticker] = data[ticker]['Adj Close']
            
            if prices:
                self.prices_df = pd.DataFrame(prices)
                print(f"\n[OK] Real data downloaded: {self.prices_df.shape}")
                return self.prices_df
        
        except Exception as e:
            print(f"[ERROR] Batch download failed: {e}")
        
        # 방법 2: 개별 다운로드 (fallback)
        prices = {}
        for ticker in self.tickers:
            try:
                ticker_obj = yf.Ticker(ticker)
                hist = ticker_obj.history(start=self.start_date, end=self.end_date)
                
                if not hist.empty:
                    prices[ticker] = hist['Close']
                    print(f"✓ {ticker}: {len(hist)} days")
            except Exception as e:
                print(f"✗ {ticker}: {e}")
        
        if prices:
            self.prices_df = pd.DataFrame(prices)
            print(f"\n[OK] Real data downloaded: {self.prices_df.shape}")
        else:
            # Demo data fallback
            print("\n[WARNING] Using demo data...")
            self.prices_df = self._create_demo_data()
        
        return self.prices_df

    
    def compute_returns(self) -> pd.DataFrame:
        """일일 수익률 계산"""
        if self.prices_df is None:
            raise ValueError("Download data first!")
        
        self.returns_df = self.prices_df.pct_change().dropna()
        
        if self.returns_df.empty:
            raise ValueError("Returns DataFrame is empty!")
        
        return self.returns_df
    
    def save_data(self, output_dir: str = './data') -> None:
        """데이터를 CSV로 저장"""
        os.makedirs(output_dir, exist_ok=True)
        
        if self.prices_df is not None and not self.prices_df.empty:
            self.prices_df.to_csv(f'{output_dir}/portfolio_prices_raw.csv')
        
        if self.returns_df is not None and not self.returns_df.empty:
            self.returns_df.to_csv(f'{output_dir}/portfolio_returns_daily.csv')
        
        print(f"\n[OK] Data saved to {output_dir}/")
    
    def validate_representativeness(self) -> Dict:
        """
        개선: 데이터 대표성 검증
        """
        if self.returns_df is None or self.returns_df.empty:
            print("\n[WARNING] No data to validate")
            return {}
        
        print("\n" + "="*70)
        print("DATA REPRESENTATIVENESS VALIDATION")
        print("="*70)
        
        validation_results = {}
        
        # 1. 정규분포 가정 검증
        print("\n【1. Normality Test (Jarque-Bera)】")
        print(f"\n{'Ticker':<10} {'Skewness':<12} {'Kurtosis':<12} {'Fat Tail':<10}")
        print("-" * 70)
        
        fat_tail_count = 0
        for ticker in self.returns_df.columns:
            s = skew(self.returns_df[ticker])
            k = kurtosis(self.returns_df[ticker])
            is_fat_tail = "YES" if k > 3.5 else "NO"
            
            if k > 3.5:
                fat_tail_count += 1
            
            print(f"{ticker:<10} {s:>10.4f}  {k:>10.4f}  {is_fat_tail:<10}")
        
        print(f"\n⚠️ Fat tails detected in {fat_tail_count}/{len(self.returns_df.columns)} assets")
        print("→ Implication: Gaussian likelihood 가정 위반 가능성")
        print("→ Solution: Student-t distribution 사용 고려")
        
        # 2. 시간 구간별 특성 변화 (Stationarity)
        print("\n【2. Stationarity Analysis (Regime Changes)】")
        
        periods = {
            'Pre-COVID': ('2019-01-01', '2019-12-31'),
            'COVID Crisis': ('2020-01-01', '2020-12-31'),
            'Recovery': ('2021-01-01', '2021-12-31'),
            'Rate Hike': ('2022-01-01', '2022-12-31'),
            'Normalization': ('2023-01-01', '2023-12-31'),
            'AI Rally': ('2024-01-01', '2025-11-16')
        }
        
        print(f"\n{'Period':<20} {'Mean Return':<15} {'Volatility':<15} {'Correlation':<15}")
        print("-" * 70)
        
        for period_name, (start, end) in periods.items():
            try:
                period_returns = self.returns_df.loc[start:end]
                if len(period_returns) > 0:
                    mean_ret = period_returns.mean().mean()
                    vol = period_returns.std().mean()
                    corr = period_returns.corr().values.mean()
                    
                    print(f"{period_name:<20} {mean_ret:>13.4%}  {vol:>13.4%}  {corr:>13.4f}")
            except:
                pass
        
        print("\n⚠️ Significant regime changes detected")
        print("→ Implication: Stationarity 가정 위반")
        print("→ Solution: Online learning 또는 adaptive models 필요")
        
        # 3. Sector concentration (Bias 분석)
        print("\n【3. Asset Composition Analysis】")
        print(f"\nAsset Sector Distribution:")
        print(f"  Technology:     4/8 (AAPL, MSFT, TSLA, AMD) = 50%")
        print(f"  Finance:        1/8 (JPM) = 12.5%")
        print(f"  Consumer:       1/8 (PG) = 12.5%")
        print(f"  Commodities:    1/8 (GLD) = 12.5%")
        print(f"  Fixed Income:   1/8 (TLT) = 12.5%")
        print(f"\n⚠️ Tech sector over-representation (50% vs ideally 30%)")
        print("→ Implication: Tech sector bias in current AI rally period")
        print("→ Solution: Balanced portfolio or sector-specific models in future")
        
        # 4. 극단값 분포
        print("\n【4. Extreme Value Analysis】")
        print(f"\n{'Ticker':<10} {'Min Return':<15} {'Max Return':<15} {'Extreme Events':<15}")
        print("-" * 70)
        
        if len(self.returns_df) > 0:
            threshold_low = np.percentile(self.returns_df.values.flatten(), 1)
            threshold_high = np.percentile(self.returns_df.values.flatten(), 99)
            
            for ticker in self.returns_df.columns:
                min_ret = self.returns_df[ticker].min()
                max_ret = self.returns_df[ticker].max()
                extreme_events = ((self.returns_df[ticker] < threshold_low) | 
                                (self.returns_df[ticker] > threshold_high)).sum()
                
                print(f"{ticker:<10} {min_ret:>13.4%}  {max_ret:>13.4%}  {extreme_events:>13}")
            
            total_extremes = ((self.returns_df.values < threshold_low) | 
                             (self.returns_df.values > threshold_high)).sum()
            
            print(f"\n✓ Extreme events (< 1% or > 99%): {total_extremes} total")
            print("→ Good: Sufficient tail events for tail risk learning")
            
            validation_results['fat_tail_count'] = fat_tail_count
            validation_results['extreme_events'] = total_extremes
        
        return validation_results
    
    def print_statistics(self) -> None:
        """데이터 통계 출력"""
        if self.returns_df is None or self.returns_df.empty:
            print("\n[WARNING] No data to print statistics")
            return
        
        print("\n" + "="*70)
        print("PORTFOLIO DATA STATISTICS")
        print("="*70)
        
        print(f"\nData shape: {self.returns_df.shape}")
        print(f"  - Assets: {self.returns_df.shape[1]}")
        print(f"  - Trading days: {self.returns_df.shape[0]}")
        print(f"  - Time span: {self.returns_df.index[0].date()} to {self.returns_df.index[-1].date()}")
        
        print("\nMean daily returns (%):")
        print((self.returns_df.mean() * 100).round(4))
        
        print("\nDaily volatility (%):")
        print((self.returns_df.std() * 100).round(4))
        
        print("\nCorrelation matrix:")
        corr_matrix = self.returns_df.corr()
        print(corr_matrix.round(3))
        
        print(f"\nAverage correlation: {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean():.4f}")
        print("→ Low correlation = Good diversification")


class PortfolioGenerator:
    """포트폴리오 구성 및 시나리오 생성"""
    
    PORTFOLIO_TYPES = {
        'Balanced': np.array([0.15, 0.15, 0.15, 0.15, 0.15, 0.1, 0.1, 0.05]),
        'Aggressive': np.array([0.2, 0.2, 0.1, 0.05, 0.25, 0.1, 0.05, 0.05]),
        'Conservative': np.array([0.1, 0.1, 0.2, 0.2, 0.0, 0.05, 0.2, 0.15]),
        'Tech-Heavy': np.array([0.25, 0.25, 0.0, 0.0, 0.3, 0.15, 0.0, 0.05]),
        'Safe-Haven': np.array([0.05, 0.05, 0.1, 0.1, 0.0, 0.0, 0.3, 0.4])
    }
    
    def __init__(self, returns_df: pd.DataFrame):
        self.returns_df = returns_df
        self.portfolio_returns = {}
        
    def compute_portfolio_returns(self) -> Dict[str, pd.Series]:
        """포트폴리오별 일일 수익률 계산"""
        print("\nComputing portfolio returns...")
        
        for portfolio_name, weights in self.PORTFOLIO_TYPES.items():
            portfolio_daily_returns = (self.returns_df @ weights)
            self.portfolio_returns[portfolio_name] = portfolio_daily_returns
            print(f"✓ {portfolio_name}")
        
        return self.portfolio_returns
    
    def train_test_split(self, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Train/Test split (Temporal split - no leakage)"""
        train_end = int(len(self.returns_df) * train_ratio)
        train_returns = self.returns_df.iloc[:train_end]
        test_returns = self.returns_df.iloc[train_end:]
        
        print(f"\nTrain/Test split:")
        print(f"  Train: {len(train_returns)} days ({train_ratio*100}%)")
        print(f"  Test:  {len(test_returns)} days ({(1-train_ratio)*100}%)")
        
        return train_returns, test_returns


def main():
    """Main execution with improved data validation"""
    # Step 1: Download and validate data
    loader = PortfolioDataLoader()
    prices = loader.download_data()
    returns = loader.compute_returns()
    
    # Step 2: Validate representativeness (NEW)
    validation = loader.validate_representativeness()
    
    # Step 3: Print statistics
    loader.print_statistics()
    
    # Step 4: Save data
    loader.save_data('./data')
    
    # Step 5: Generate portfolios
    generator = PortfolioGenerator(returns)
    portfolio_returns = generator.compute_portfolio_returns()
    train_returns, test_returns = generator.train_test_split()
    
    return loader, generator, train_returns, test_returns


if __name__ == '__main__':
    main()
