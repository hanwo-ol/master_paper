import pandas as pd
import numpy as np

# 검증 결과 로드
validation_df = pd.read_csv('./validation_results.csv')

print("=" * 80)
print("Feature 누락 분석")
print("=" * 80)

# 첫 번째 종목의 누락된 feature 확인
first_ticker = validation_df.iloc[0]
print(f"\n첫 번째 종목: {first_ticker['ticker']}")
print(f"총 feature 수: {first_ticker['total_features']}")
print(f"누락된 feature: {first_ticker['missing_features']}")

# 실제 데이터 로드해서 확인
sample_ticker = validation_df.iloc[0]['ticker']
sample_data = pd.read_csv(f'./sp500_data/tickers/{sample_ticker}.csv', index_col=0)

print(f"\n실제 컬럼 목록 ({len(sample_data.columns)}개):")
print(sample_data.columns.tolist())

# 기대하는 40개 feature
expected_features = [
    # Price-based (5)
    'open', 'high', 'low', 'close', 'volume',
    # Technical (18)
    'rsi', 'ma_20', 'ma_50', 'ma_200', 'macd', 'macd_signal',
    'bollinger_upper', 'bollinger_lower', 'atr', 'stoch_k', 'stoch_d',
    'price_roc', 'realized_vol_20', 'realized_vol_60', 'volume_roc',
    'mfi', 'obv', 'williams_r',
    # Fundamental (10)
    'pe_ratio', 'pb_ratio', 'roe', 'roa', 'debt_to_equity',
    'dividend_yield', 'payout_ratio', 'free_cashflow_yield', 'asset_turnover',
    # Macro (7)
    'vix', 'treasury_10y', 'treasury_2y', 'yield_spread',
    'usd_index', 'credit_spread', 'cpi_yoy'
]

# 누락된 feature 찾기
actual_features = [col for col in sample_data.columns if col != 'returns']
missing = set(expected_features) - set(actual_features)
extra = set(actual_features) - set(expected_features)

print(f"\n누락된 feature ({len(missing)}개):")
for f in missing:
    print(f"  - {f}")

print(f"\n추가된 feature ({len(extra)}개):")
for f in extra:
    print(f"  - {f}")

# 각 카테고리별 feature 수 확인
categories = {
    'Price-based': ['open', 'high', 'low', 'close', 'volume'],
    'Technical': ['rsi', 'ma_20', 'ma_50', 'ma_200', 'macd', 'macd_signal',
                  'bollinger_upper', 'bollinger_lower', 'atr', 'stoch_k', 'stoch_d',
                  'price_roc', 'realized_vol_20', 'realized_vol_60', 'volume_roc',
                  'mfi', 'obv', 'williams_r'],
    'Fundamental': ['pe_ratio', 'pb_ratio', 'roe', 'roa', 'debt_to_equity',
                    'dividend_yield', 'payout_ratio', 'free_cashflow_yield', 'asset_turnover'],
    'Macro': ['vix', 'treasury_10y', 'treasury_2y', 'yield_spread',
              'usd_index', 'credit_spread', 'cpi_yoy']
}

print("\n카테고리별 feature 존재 여부:")
for category, features in categories.items():
    present = sum(1 for f in features if f in actual_features)
    print(f"  {category}: {present}/{len(features)}")
    missing_in_cat = [f for f in features if f not in actual_features]
    if missing_in_cat:
        print(f"    누락: {missing_in_cat}")

# Panel data 확인
print("\n\nPanel Data 정보:")
panel_data = pd.read_csv('./sp500_data/sp500_panel.csv', index_col=0, nrows=5)
print(f"Panel data 컬럼 ({len(panel_data.columns)}개):")
print(panel_data.columns.tolist())

# 데이터 품질 확인
print("\n\n데이터 품질 요약:")
print(f"총 종목 수: 501")
print(f"총 데이터 포인트: 2,406,597")
print(f"평균 기간: {4804 / 252:.1f} 년")
print(f"실패한 티커: Q, SOLS")
