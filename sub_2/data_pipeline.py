"""
data_pipeline.py

- S&P 500 티커 수집
- 2005-01-01 ~ 2025-12-31 주가(OHLCV) 수집
- Parquet 포맷으로 로컬 저장

주의:
- 위키피디아에서 403이 뜰 수 있으므로, User-Agent 헤더를 사용.
- 그래도 안 되면 로컬 'sp500_tickers.csv' 파일에서 티커를 읽도록 fallback.
"""

import os
import time
import logging
from pathlib import Path
from typing import List

import pandas as pd
import yfinance as yf
import requests


# ---------------------------------------
# 기본 설정
# ---------------------------------------

START_DATE = "2005-01-01"
END_DATE = "2025-12-31"

# 현재 스크립트가 있는 디렉토리 기준으로 저장
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

TICKER_CSV_PATH = DATA_DIR / "sp500_tickers.csv"
EQUITY_PARQUET_PATH = DATA_DIR / "equity_ohlcv_2005_2025.parquet"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# ---------------------------------------
# 1. S&P 500 티커 가져오기
# ---------------------------------------

def get_sp500_tickers_from_wikipedia() -> List[str]:
    """
    위키피디아에서 S&P 500 구성종목 티커를 읽는다.
    User-Agent 헤더를 사용해서 403을 피하려고 시도.
    실패시 예외를 던진다.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    logging.info(f"Fetching S&P 500 tickers from Wikipedia: {url}")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()  # 4xx/5xx면 예외 발생

    # HTML 문자열에 대해 read_html 수행
    tables = pd.read_html(resp.text)
    df = tables[0]

    # 위키 구조가 변해도 보통 'Symbol' 컬럼이 있음
    if "Symbol" not in df.columns:
        raise ValueError("Could not find 'Symbol' column in Wikipedia table.")

    tickers = df["Symbol"].astype(str).str.strip().tolist()

    # BRK.B -> BRK-B 형식으로 변환 (yfinance 스타일)
    tickers = [t.replace(".", "-") for t in tickers]

    logging.info(f"Fetched {len(tickers)} tickers from Wikipedia.")
    return tickers


def get_sp500_tickers_from_local() -> List[str]:
    """
    로컬 CSV(TICKER_CSV_PATH)에 저장된 S&P 500 티커 목록을 읽는다.
    CSV는 최소한 'Symbol' 또는 'Ticker' 컬럼을 포함해야 한다.
    """
    if not TICKER_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Local ticker file not found: {TICKER_CSV_PATH}\n"
            "If Wikipedia access keeps failing, create this file manually.\n"
            "Example format:\n"
            "Symbol\n"
            "AAPL\n"
            "MSFT\n"
            "...\n"
        )

    logging.info(f"Loading S&P 500 tickers from local file: {TICKER_CSV_PATH}")
    df = pd.read_csv(TICKER_CSV_PATH)

    col = None
    for candidate in ["Symbol", "Ticker", "symbol", "ticker"]:
        if candidate in df.columns:
            col = candidate
            break
    if col is None:
        raise ValueError(
            f"Local ticker file {TICKER_CSV_PATH} must contain a 'Symbol' or 'Ticker' column."
        )

    tickers = df[col].astype(str).str.strip().tolist()
    tickers = [t.replace(".", "-") for t in tickers]
    logging.info(f"Loaded {len(tickers)} tickers from local CSV.")
    return tickers


def get_sp500_tickers() -> List[str]:
    """
    1차: Wikipedia 시도 (User-Agent 포함)
    2차: 실패하면 로컬 sp500_tickers.csv 파일 사용
    """
    try:
        return get_sp500_tickers_from_wikipedia()
    except Exception as e:
        logging.warning(f"Failed to fetch tickers from Wikipedia: {e}")
        logging.warning("Falling back to local CSV: sp500_tickers.csv")
        return get_sp500_tickers_from_local()


# ---------------------------------------
# 2. yfinance로 OHLCV 다운로드
# ---------------------------------------

def download_equity_data(tickers: List[str]) -> pd.DataFrame:
    """
    yfinance를 이용해 2005~2025 OHLCV 데이터를 다운로드하고
    MultiIndex (Date, Ticker) 형태의 DataFrame으로 반환.
    """

    logging.info(
        f"Downloading OHLCV data for {len(tickers)} tickers "
        f"from {START_DATE} to {END_DATE} via yfinance..."
    )

    # yfinance는 리스트/공백 구분 문자열 모두 허용
    # auto_adjust=True : Adjusted Close 기준으로 OHLC를 조정
    data = yf.download(
        tickers=tickers,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=True,
    )

    # data 구조:
    #   - tickers가 여러 개면 columns: MultiIndex (Ticker, Field)
    #   - 단일 ticker면 columns: 단일 Index(Field)
    if isinstance(data.columns, pd.MultiIndex):
        # (Date, Ticker) 인덱스로 변환
        stacked = (
            data.stack(level=0)
                .rename_axis(index=["Date", "Ticker"])
                .reset_index()
        )
    else:
        # 단일 티커일 경우: Ticker 컬럼 하나 추가
        ticker = tickers[0]
        tmp = data.copy()
        tmp["Ticker"] = ticker
        tmp = tmp.reset_index().rename(columns={"index": "Date"})
        stacked = tmp.set_index(["Date", "Ticker"])

    # 열 이름 통일
    # Open, High, Low, Close, Adj Close, Volume 중 존재하는 것만 사용
    stacked = stacked.set_index(["Date", "Ticker"])
    stacked = stacked.sort_index()

    logging.info(
        f"Downloaded data shape: {stacked.shape} "
        f"(rows x columns), index levels: {stacked.index.names}"
    )

    return stacked


# ---------------------------------------
# 3. 저장 함수
# ---------------------------------------

def save_to_parquet(df: pd.DataFrame, path: Path) -> None:
    """
    주어진 DataFrame을 Parquet 포맷으로 저장.
    """
    logging.info(f"Saving DataFrame to Parquet: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    logging.info("Saving done.")


# ---------------------------------------
# 4. 메인 파이프라인
# ---------------------------------------

def main():
    logging.info("=== Step 1: Get S&P 500 tickers ===")
    tickers = get_sp500_tickers()
    logging.info(f"Using {len(tickers)} tickers.")

    logging.info("=== Step 2: Download OHLCV data via yfinance ===")
    start_time = time.time()
    df_equity = download_equity_data(tickers)
    elapsed = time.time() - start_time
    logging.info(f"Download completed. Elapsed: {elapsed:.1f} sec")

    logging.info("=== Step 3: Save to Parquet ===")
    save_to_parquet(df_equity, EQUITY_PARQUET_PATH)

    logging.info("All done.")
    logging.info(f"Equity OHLCV saved at: {EQUITY_PARQUET_PATH}")


if __name__ == "__main__":
    main()
