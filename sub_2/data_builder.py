import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import time
import logging
from fredapi import Fred
from dotenv import load_dotenv
import os
import sys

# .env 파일에서 API 키 로드
load_dotenv('C:/Users/11015/Documents/master_paper/sub_2/.env')
FRED_API_KEY = os.getenv('FRED_API_KEY')

# 로깅 설정 (UTF-8 인코딩 강제)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_collection.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# StreamHandler의 인코딩을 UTF-8로 설정
for handler in logging.getLogger().handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.stream = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

class SNP500DataCollector:
    """
    S&P 500 데이터 수집 및 Feature Engineering 클래스
    논문에서 정의한 40개 feature를 수집합니다.
    """
    
    def __init__(self, start_date='2005-01-01', end_date='2025-11-23', ticker_csv_path=None):
        self.start_date = start_date
        self.end_date = end_date
        self.tickers = []
        self.data = {}
        self.features_df = None
        self.fred = None
        self.ticker_csv_path = ticker_csv_path
        
        # FRED API 초기화
        if FRED_API_KEY:
            self.fred = Fred(api_key=FRED_API_KEY)
            logging.info("FRED API 연결 성공")
        else:
            logging.warning("FRED API 키가 없습니다. Macro 데이터 수집이 제한됩니다.")
        
        logging.info(f"Data Collector 초기화: {start_date} ~ {end_date}")
    
    def load_tickers_from_csv(self, csv_path):
        """CSV 파일에서 S&P 500 티커 리스트 로드"""
        logging.info(f"S&P 500 티커 CSV 로드 시작: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            
            # 컬럼명 확인
            if 'Ticker' not in df.columns:
                logging.error(f"CSV 파일에 'Ticker' 컬럼이 없습니다. 컬럼: {df.columns.tolist()}")
                raise ValueError("Missing 'Ticker' column")
            
            tickers = df['Ticker'].tolist()
            
            # 티커 형식 수정 (예: BRK.B -> BRK-B)
            tickers = [ticker.replace('.', '-') for ticker in tickers]
            
            self.tickers = tickers
            logging.info(f"총 {len(tickers)}개 티커 로드 완료")
            logging.info(f"첫 10개 티커: {tickers[:10]}")
            
            return tickers
            
        except Exception as e:
            logging.error(f"CSV 티커 로드 실패: {e}")
            raise
    
    def get_sp500_tickers(self):
        """S&P 500 티커 리스트 가져오기 (CSV 또는 대체 방법)"""
        
        # Method 1: CSV 파일 사용
        if self.ticker_csv_path and os.path.exists(self.ticker_csv_path):
            try:
                return self.load_tickers_from_csv(self.ticker_csv_path)
            except Exception as e:
                logging.warning(f"CSV 로드 실패, 대체 방법 시도: {e}")
        
        # Method 2: 하드코딩된 주요 종목 리스트 사용
        logging.info("하드코딩된 주요 종목 리스트 사용...")
        
        major_tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH',
            'JNJ', 'V', 'XOM', 'WMT', 'JPM', 'MA', 'PG', 'CVX', 'HD', 'MRK',
            'ABBV', 'KO', 'PEP', 'AVGO', 'COST', 'TMO', 'MCD', 'CSCO', 'ACN', 'ABT',
            'DHR', 'LIN', 'ADBE', 'NKE', 'VZ', 'NEE', 'TXN', 'CRM', 'WFC', 'PM',
            'RTX', 'NFLX', 'UPS', 'INTC', 'HON', 'MS', 'ORCL', 'LOW', 'QCOM', 'IBM',
            'BA', 'AMD', 'UNP', 'INTU', 'CAT', 'GS', 'AMGN', 'SBUX', 'GE', 'AXP',
            'BLK', 'DE', 'SPGI', 'BKNG', 'ISRG', 'TJX', 'GILD', 'MMC', 'ADI', 'ADP',
            'CVS', 'C', 'MDLZ', 'PLD', 'CI', 'ZTS', 'SYK', 'REGN', 'SO', 'CB',
            'DUK', 'NOW', 'VRTX', 'MO', 'PGR', 'BDX', 'TGT', 'CL', 'EOG', 'EQIX',
            'ITW', 'BSX', 'APD', 'SHW', 'NOC', 'CME', 'AON', 'SCHW', 'USB', 'MMM'
        ]
        
        self.tickers = major_tickers
        logging.info(f"대체 리스트 사용: {len(major_tickers)}개 주요 종목")
        
        return major_tickers
    
    def download_price_data(self, ticker):
        """개별 종목의 OHLCV 데이터 다운로드"""
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=self.start_date, end=self.end_date)
            
            if df.empty:
                logging.warning(f"{ticker}: 데이터 없음")
                return None
            
            # Timezone 제거 (UTC -> naive)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            # 컬럼명 표준화
            df.columns = [col.lower() for col in df.columns]
            df = df.rename(columns={
                'dividends': 'dividend',
                'stock splits': 'stock_split'
            })
            
            logging.info(f"{ticker}: {len(df)}개 데이터 포인트 수집 완료")
            return df
            
        except Exception as e:
            logging.error(f"{ticker} 다운로드 실패: {e}")
            return None
    
    def download_fundamental_data(self, ticker):
        """개별 종목의 fundamental 데이터 다운로드"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # 필요한 fundamental 지표 추출
            fundamentals = {
                'pe_ratio': info.get('trailingPE', np.nan),
                'pb_ratio': info.get('priceToBook', np.nan),
                'roe': info.get('returnOnEquity', np.nan),
                'roa': info.get('returnOnAssets', np.nan),
                'debt_to_equity': info.get('debtToEquity', np.nan),
                'dividend_yield': info.get('dividendYield', np.nan),
                'payout_ratio': info.get('payoutRatio', np.nan),
                'free_cashflow_yield': info.get('freeCashflow', np.nan),
                'asset_turnover': info.get('assetTurnover', np.nan),
            }
            
            return fundamentals
            
        except Exception as e:
            logging.debug(f"{ticker} fundamental 데이터 다운로드 실패: {e}")
            return None
    
    def calculate_technical_indicators(self, df, ticker):
        """
        Technical Indicators 계산 (18개)
        논문의 Feature Engineering 섹션에 명시된 지표들
        """
        try:
            close = df['close'].copy()
            high = df['high'].copy()
            low = df['low'].copy()
            volume = df['volume'].copy()
            
            result = pd.DataFrame(index=df.index)
            
            # 1. RSI (Relative Strength Index)
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            result['rsi'] = 100 - (100 / (1 + rs))
            
            # 2-4. Moving Averages (20, 50, 200 day)
            result['ma_20'] = close.rolling(window=20).mean()
            result['ma_50'] = close.rolling(window=50).mean()
            result['ma_200'] = close.rolling(window=200).mean()
            
            # 5-6. MACD (Moving Average Convergence Divergence)
            exp1 = close.ewm(span=12, adjust=False).mean()
            exp2 = close.ewm(span=26, adjust=False).mean()
            result['macd'] = exp1 - exp2
            result['macd_signal'] = result['macd'].ewm(span=9, adjust=False).mean()
            
            # 7-8. Bollinger Bands (20-day)
            ma20 = close.rolling(window=20).mean()
            std20 = close.rolling(window=20).std()
            result['bollinger_upper'] = ma20 + (std20 * 2)
            result['bollinger_lower'] = ma20 - (std20 * 2)
            
            # 9. ATR (Average True Range)
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            result['atr'] = tr.rolling(window=14).mean()
            
            # 10-11. Stochastic Oscillator
            low_min = low.rolling(window=14).min()
            high_max = high.rolling(window=14).max()
            result['stoch_k'] = 100 * (close - low_min) / (high_max - low_min)
            result['stoch_d'] = result['stoch_k'].rolling(window=3).mean()
            
            # 12. Price Rate of Change
            result['price_roc'] = close.pct_change(periods=10) * 100
            
            # 13-14. Realized Volatility (20 and 60 day)
            returns = close.pct_change()
            result['realized_vol_20'] = returns.rolling(window=20).std() * np.sqrt(252)
            result['realized_vol_60'] = returns.rolling(window=60).std() * np.sqrt(252)
            
            # 15. Volume Rate of Change
            result['volume_roc'] = volume.pct_change(periods=10) * 100
            
            # 16. Money Flow Index (MFI)
            typical_price = (high + low + close) / 3
            money_flow = typical_price * volume
            positive_flow = money_flow.where(typical_price > typical_price.shift(), 0).rolling(window=14).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(), 0).rolling(window=14).sum()
            mfi_ratio = positive_flow / negative_flow
            result['mfi'] = 100 - (100 / (1 + mfi_ratio))
            
            # 17. On-Balance Volume (OBV)
            obv = [0]
            for i in range(1, len(close)):
                if close.iloc[i] > close.iloc[i-1]:
                    obv.append(obv[-1] + volume.iloc[i])
                elif close.iloc[i] < close.iloc[i-1]:
                    obv.append(obv[-1] - volume.iloc[i])
                else:
                    obv.append(obv[-1])
            result['obv'] = obv
            
            # 18. Williams %R
            result['williams_r'] = -100 * (high_max - close) / (high_max - low_min)
            
            logging.info(f"{ticker}: Technical indicators 계산 완료")
            return result
            
        except Exception as e:
            logging.error(f"{ticker} Technical indicators 계산 실패: {e}")
            return None
    
    def download_macro_data(self):
        """
        Macroeconomic Indicators 다운로드 (7개)
        FRED API를 사용하여 실제 macro 데이터 수집
        """
        logging.info("Macroeconomic 데이터 다운로드 시작...")
        
        try:
            macro_data = pd.DataFrame()
            
            # 1. VIX Index (Yahoo Finance)
            logging.info("  - VIX 다운로드 중...")
            try:
                vix = yf.download('^VIX', start=self.start_date, end=self.end_date, progress=False)
                if not vix.empty:
                    if vix.index.tz is not None:
                        vix.index = vix.index.tz_localize(None)
                    macro_data['vix'] = vix['Close']
                    logging.info(f"    [OK] VIX: {len(vix)} rows")
            except Exception as e:
                logging.warning(f"    [FAIL] VIX: {e}")
            
            # 2. 10-Year Treasury Yield (Yahoo Finance)
            logging.info("  - 10Y Treasury Yield 다운로드 중...")
            try:
                tnx = yf.download('^TNX', start=self.start_date, end=self.end_date, progress=False)
                if not tnx.empty:
                    if tnx.index.tz is not None:
                        tnx.index = tnx.index.tz_localize(None)
                    macro_data['treasury_10y'] = tnx['Close']
                    logging.info(f"    [OK] 10Y Treasury: {len(tnx)} rows")
            except Exception as e:
                logging.warning(f"    [FAIL] 10Y Treasury: {e}")
            
            # 3. 2-Year Treasury Yield (FRED)
            if self.fred:
                logging.info("  - 2Y Treasury Yield (FRED) 다운로드 중...")
                try:
                    dgs2 = self.fred.get_series('DGS2', observation_start=self.start_date, observation_end=self.end_date)
                    macro_data['treasury_2y'] = dgs2
                    logging.info(f"    [OK] 2Y Treasury: {len(dgs2)} rows")
                except Exception as e:
                    logging.warning(f"    [FAIL] 2Y Treasury: {e}")
                    macro_data['treasury_2y'] = np.nan
            
            # 4. Treasury Yield Spread (10Y - 2Y)
            if 'treasury_10y' in macro_data.columns and 'treasury_2y' in macro_data.columns:
                macro_data['yield_spread'] = macro_data['treasury_10y'] - macro_data['treasury_2y']
                logging.info("    [OK] Yield Spread 계산 완료")
            else:
                macro_data['yield_spread'] = np.nan
                logging.warning("    [FAIL] Yield Spread 계산 실패")
            
            # 5. USD Index (Yahoo Finance)
            logging.info("  - USD Index 다운로드 중...")
            try:
                usd = yf.download('DX-Y.NYB', start=self.start_date, end=self.end_date, progress=False)
                if not usd.empty:
                    if usd.index.tz is not None:
                        usd.index = usd.index.tz_localize(None)
                    macro_data['usd_index'] = usd['Close']
                    logging.info(f"    [OK] USD Index: {len(usd)} rows")
            except Exception as e:
                logging.warning(f"    [FAIL] USD Index: {e}")
                macro_data['usd_index'] = np.nan
            
            # 6. Credit Spread (FRED: BAA - AAA Corporate Bond Yields)
            if self.fred:
                logging.info("  - Credit Spread (FRED) 다운로드 중...")
                try:
                    baa = self.fred.get_series('DBAA', observation_start=self.start_date, observation_end=self.end_date)
                    aaa = self.fred.get_series('DAAA', observation_start=self.start_date, observation_end=self.end_date)
                    credit_spread = baa - aaa
                    macro_data['credit_spread'] = credit_spread
                    logging.info(f"    [OK] Credit Spread: {len(credit_spread)} rows")
                except Exception as e:
                    logging.warning(f"    [FAIL] Credit Spread: {e}")
                    macro_data['credit_spread'] = np.nan
            
            # 7. CPI Year-over-Year (FRED)
            if self.fred:
                logging.info("  - CPI YoY (FRED) 다운로드 중...")
                try:
                    cpi = self.fred.get_series('CPIAUCSL', observation_start=self.start_date, observation_end=self.end_date)
                    cpi_yoy = cpi.pct_change(periods=12) * 100
                    macro_data['cpi_yoy'] = cpi_yoy
                    logging.info(f"    [OK] CPI YoY: {len(cpi_yoy)} rows")
                except Exception as e:
                    logging.warning(f"    [FAIL] CPI YoY: {e}")
                    macro_data['cpi_yoy'] = np.nan
            else:
                macro_data['cpi_yoy'] = np.nan
                logging.warning("    [FAIL] CPI YoY: FRED API 없음")
            
            # Forward fill for missing values
            macro_data = macro_data.fillna(method='ffill')
            
            # 누락 비율 체크
            missing_pct = (macro_data.isnull().sum() / len(macro_data) * 100)
            logging.info("\n  [Macro Data 품질 체크]")
            for col in macro_data.columns:
                pct = missing_pct[col]
                status = "[OK]" if pct < 5 else "[WARN]" if pct < 20 else "[FAIL]"
                logging.info(f"    {status} {col}: {pct:.1f}% missing")
            
            logging.info(f"\nMacro 데이터 수집 완료: {len(macro_data)} rows, {len(macro_data.columns)} columns")
            return macro_data
            
        except Exception as e:
            logging.error(f"Macro 데이터 다운로드 실패: {e}")
            return None
    
    def construct_feature_tensor(self, ticker):
        """
        개별 종목의 40개 feature 구축
        """
        # 1. Price data download
        price_df = self.download_price_data(ticker)
        if price_df is None or len(price_df) < 252:
            logging.warning(f"{ticker}: 충분한 데이터 없음 (< 252 days)")
            return None
        
        # 2. Technical indicators (18개)
        tech_indicators = self.calculate_technical_indicators(price_df, ticker)
        if tech_indicators is None:
            return None
        
        # 3. Price-based features (5개: OHLCV)
        features = pd.DataFrame(index=price_df.index)
        features['open'] = price_df['open']
        features['high'] = price_df['high']
        features['low'] = price_df['low']
        features['close'] = price_df['close']
        features['volume'] = price_df['volume']
        
        # 4. Technical indicators 추가 (18개)
        features = pd.concat([features, tech_indicators], axis=1)
        
        # 5. Fundamental features (10개)
        fundamentals = self.download_fundamental_data(ticker)
        if fundamentals:
            for key, value in fundamentals.items():
                features[key] = value
        else:
            for key in ['pe_ratio', 'pb_ratio', 'roe', 'roa', 'debt_to_equity',
                       'dividend_yield', 'payout_ratio', 'free_cashflow_yield', 'asset_turnover']:
                features[key] = np.nan
        
        # Forward fill fundamental data
        features = features.fillna(method='ffill')
        
        # 6. Returns 계산 (target variable)
        features['returns'] = features['close'].pct_change()
        
        # Feature 개수 확인
        num_features = len([col for col in features.columns if col != 'returns'])
        logging.info(f"{ticker}: Feature tensor 구축 완료 - Shape: {features.shape}, Features: {num_features}")
        
        return features
    
    def collect_all_data(self, max_tickers=None):
        """
        모든 S&P 500 종목 데이터 수집
        """
        logging.info("=" * 80)
        logging.info("전체 데이터 수집 시작")
        logging.info("=" * 80)
        
        # 1. 티커 리스트 가져오기
        if not self.tickers:
            self.get_sp500_tickers()
        
        if max_tickers:
            self.tickers = self.tickers[:max_tickers]
            logging.info(f"테스트 모드: {max_tickers}개 종목만 수집")
        
        # 2. Macro data 다운로드 (전체 종목 공통)
        macro_data = self.download_macro_data()
        
        # 3. 개별 종목 데이터 수집
        successful = 0
        failed = 0
        failed_tickers = []
        
        for ticker in tqdm(self.tickers, desc="Collecting data"):
            try:
                features = self.construct_feature_tensor(ticker)
                
                if features is not None:
                    # Macro data 병합
                    if macro_data is not None:
                        # Timezone 확인
                        if features.index.tz is not None:
                            features.index = features.index.tz_localize(None)
                        if macro_data.index.tz is not None:
                            macro_data.index = macro_data.index.tz_localize(None)
                        
                        features = features.join(macro_data, how='left')
                        features = features.fillna(method='ffill')
                    
                    self.data[ticker] = features
                    successful += 1
                else:
                    failed += 1
                    failed_tickers.append(ticker)
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logging.error(f"{ticker}: 예상치 못한 오류 - {e}")
                failed += 1
                failed_tickers.append(ticker)
        
        logging.info("=" * 80)
        logging.info(f"데이터 수집 완료: 성공 {successful}, 실패 {failed}")
        if failed_tickers:
            logging.info(f"실패한 티커: {', '.join(failed_tickers[:10])}{'...' if len(failed_tickers) > 10 else ''}")
        logging.info("=" * 80)
        
        return self.data
    
    def validate_features(self):
        """
        수집된 feature의 유효성 검증
        """
        logging.info("=" * 80)
        logging.info("Feature 유효성 검증 시작")
        logging.info("=" * 80)
        
        if len(self.data) == 0:
            logging.error("수집된 데이터가 없습니다!")
            return pd.DataFrame()
        
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
        
        validation_results = []
        
        for ticker, df in self.data.items():
            missing_features = [f for f in expected_features if f not in df.columns]
            missing_pct = (df.isnull().sum() / len(df) * 100).to_dict()
            
            actual_features = [col for col in df.columns if col != 'returns']
            
            validation_results.append({
                'ticker': ticker,
                'total_features': len(actual_features),
                'has_all_40': len(missing_features) == 0 and len(actual_features) >= 40,
                'missing_features': missing_features,
                'date_range': f"{df.index.min()} to {df.index.max()}",
                'total_days': len(df)
            })
        
        validation_df = pd.DataFrame(validation_results)
        
        # 요약 통계
        logging.info(f"\n총 {len(validation_df)}개 종목 검증 완료")
        logging.info(f"평균 feature 수: {validation_df['total_features'].mean():.1f}")
        logging.info(f"40개 feature 완비 종목: {validation_df['has_all_40'].sum()}/{len(validation_df)}")
        logging.info(f"평균 데이터 기간: {validation_df['total_days'].mean():.0f} days")
        
        # 검증 결과 저장
        validation_df.to_csv('./validation_results.csv', index=False)
        logging.info("\n검증 결과 저장: ./validation_results.csv")
        
        return validation_df
    
    def save_data(self, output_dir='./sp500_data'):
        """
        수집된 데이터를 파일로 저장
        """
        os.makedirs(output_dir, exist_ok=True)
        
        logging.info(f"데이터 저장 시작: {output_dir}")
        
        # 1. 개별 종목 데이터 저장 (CSV)
        ticker_dir = os.path.join(output_dir, 'tickers')
        os.makedirs(ticker_dir, exist_ok=True)
        
        for ticker, df in self.data.items():
            filepath = os.path.join(ticker_dir, f'{ticker}.csv')
            df.to_csv(filepath)
        
        # 2. Consolidated panel data 저장
        panel_data = []
        for ticker, df in self.data.items():
            df_copy = df.copy()
            df_copy['ticker'] = ticker
            panel_data.append(df_copy)
        
        panel_df = pd.concat(panel_data, axis=0)
        panel_df.to_csv(os.path.join(output_dir, 'sp500_panel.csv'))
        
        # 3. 메타데이터 저장
        metadata = {
            'collection_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'start_date': self.start_date,
            'end_date': self.end_date,
            'num_tickers': len(self.data),
            'tickers': list(self.data.keys()),
            'fred_api_used': self.fred is not None
        }
        
        import json
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logging.info(f"데이터 저장 완료: {output_dir}")
        logging.info(f"  - {len(self.data)}개 종목 CSV 파일")
        logging.info(f"  - Panel data shape: {panel_df.shape}")


# 실행 예제
if __name__ == "__main__":
    print("S&P 500 Data Collection Script (CSV-based)")
    print("=" * 80)
    
    # FRED API 키 확인
    if FRED_API_KEY:
        print(f"[OK] FRED API 키 로드 완료: {FRED_API_KEY[:10]}...")
    else:
        print("[WARN] FRED API 키를 찾을 수 없습니다.")
    
    # CSV 티커 파일 경로
    ticker_csv_path = 'C:/Users/11015/Documents/master_paper/sub_2/data/sp500_tickers.csv'
    
    # 데이터 수집기 초기화
    collector = SNP500DataCollector(
        start_date='2005-01-01',
        end_date='2025-11-23',
        ticker_csv_path=ticker_csv_path
    )
    
    # 옵션 선택
    print("\n[옵션 선택]")
    print("1. 테스트 모드 (10개 종목, ~10분)")
    print("2. 중간 모드 (50개 종목, ~30분)")
    print("3. 전체 수집 (500+ 종목, ~3시간)")
    
    choice = input("선택 (1, 2, 또는 3): ")
    
    if choice == '1':
        max_tickers = 10
    elif choice == '2':
        max_tickers = 50
    else:
        max_tickers = None
    
    # 데이터 수집
    data = collector.collect_all_data(max_tickers=max_tickers)
    
    # 유효성 검증
    if len(data) > 0:
        validation_results = collector.validate_features()
        
        # 데이터 저장
        collector.save_data(output_dir='./sp500_data')
        
        print("\n" + "=" * 80)
        print("데이터 수집 완료!")
        print(f"수집된 종목 수: {len(data)}")
        print("저장 위치: ./sp500_data/")
        print("검증 결과: ./validation_results.csv")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("[ERROR] 데이터 수집 실패!")
        print("=" * 80)
