#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
v1 데이터 수집기:
- Yahoo Finance 주가/거래량 (OHLCV) 다운로드 + 캐싱
- FRED 매크로 지표 다운로드 + 캐싱

사용 예시:
    python data_collector_v1.py \
        --tickers-file tickers.txt \
        --start 2000-01-01 \
        --end 2024-12-31 \
        --fred-series DGS10 T10Y2Y VIXCLS

tickers.txt 예시:
    AAPL
    MSFT
    AMZN
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional, Dict

import pandas as pd
import yfinance as yf
from fredapi import Fred

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv 안 쓸 거면 무시 가능
    pass


# -------------------------------------------------------------------
# 경로 설정
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
YAHOO_DIR = DATA_DIR / "yahoo"
FRED_DIR = DATA_DIR / "fred"

for d in [DATA_DIR, YAHOO_DIR, FRED_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------
# 유틸 함수
# -------------------------------------------------------------------

def parse_date(date_str: str) -> pd.Timestamp:
    return pd.Timestamp(date_str).tz_localize(None)


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize(None)
    return df.sort_index()


# -------------------------------------------------------------------
# Yahoo Finance 다운로드 모듈
# -------------------------------------------------------------------

def load_tickers_from_file(path: Path) -> List[str]:
    """
    CSV 또는 TXT에서 티커 리스트를 읽어온다.
    - .csv : 첫 번째 컬럼을 티커로 사용
    - .txt : 줄 단위 티커
    """
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        # 1) "ticker" 라는 컬럼명이 있으면 그걸 쓰고,
        # 2) 없으면 첫 번째 컬럼을 사용
        if "ticker" in df.columns:
            tickers = df["ticker"].astype(str).str.strip().tolist()
        else:
            first_col = df.columns[0]
            tickers = df[first_col].astype(str).str.strip().tolist()
        # 빈 문자열 제거
        tickers = [t for t in tickers if t]
        return tickers

    # 기존 txt 방식도 그대로 살려두기
    tickers = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                tickers.append(line)
    return tickers


def get_existing_date_range(file_path: Path) -> Optional[pd.DatetimeIndex]:
    if not file_path.exists():
        return None
    try:
        df = pd.read_parquet(file_path)
        df = ensure_datetime_index(df)
        return df.index
    except Exception:
        return None


def download_yahoo_for_ticker(
    ticker: str,
    start: str,
    end: str,
    force: bool = False,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    개별 티커에 대해:
    - 이미 로컬에 있는 데이터를 읽어서
    - 부족한 구간만 Yahoo에서 추가로 받아 병합
    - 최종 결과를 parquet로 저장
    """
    file_path = YAHOO_DIR / f"{ticker}.parquet"
    start_dt = parse_date(start)
    end_dt = parse_date(end)

    existing_df = None
    existing_range = get_existing_date_range(file_path)
    if existing_range is not None and not force:
        existing_df = pd.read_parquet(file_path)
        existing_df = ensure_datetime_index(existing_df)

    # 새로 받을 범위 결정
    download_start = start_dt
    if existing_range is not None and not force:
        # 기존 데이터의 마지막 날짜 이후만 받기
        last_date = existing_range.max()
        if last_date >= end_dt:
            # 이미 원하는 범위를 모두 가지고 있음
            print(f"[Yahoo] {ticker}: already up-to-date ({file_path})")
            return existing_df.loc[(existing_df.index >= start_dt) & (existing_df.index <= end_dt)]
        else:
            # 마지막 날짜 다음 날부터 새로 받기
            download_start = last_date + pd.Timedelta(days=1)

    print(f"[Yahoo] Download {ticker}: {download_start.date()} → {end_dt.date()}")
    new_df = yf.download(
        ticker,
        start=download_start.strftime("%Y-%m-%d"),
        end=(end_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),  # yfinance end is exclusive
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    if new_df.empty and existing_df is None:
        print(f"[Yahoo] WARNING: no data downloaded for {ticker}")
        return pd.DataFrame()

    if not new_df.empty:
        new_df = ensure_datetime_index(new_df)

    if existing_df is not None and not existing_df.empty and not force:
        combined = pd.concat([existing_df, new_df])
        combined = combined[~combined.index.duplicated(keep="first")]
    else:
        combined = new_df

    combined = ensure_datetime_index(combined)
    # 전체를 저장
    combined.to_parquet(file_path)
    print(f"[Yahoo] Saved {ticker} to {file_path} (rows={len(combined)})")

    # 요청한 기간만 잘라서 반환
    return combined.loc[(combined.index >= start_dt) & (combined.index <= end_dt)]


def download_yahoo_prices(
    tickers: List[str],
    start: str,
    end: str,
    force: bool = False,
    interval: str = "1d",
) -> Dict[str, pd.DataFrame]:
    """
    여러 티커에 대해 Yahoo 가격 데이터 다운로드 + 캐싱.
    반환값: {ticker: DataFrame}
    """
    result = {}
    for t in tickers:
        df = download_yahoo_for_ticker(t, start, end, force=force, interval=interval)
        result[t] = df
    return result


# -------------------------------------------------------------------
# FRED 매크로 시계열 다운로드 모듈
# -------------------------------------------------------------------

def get_fred_client(api_key: Optional[str] = None) -> Fred:
    if api_key is None:
        api_key = os.getenv("FRED_API_KEY", None)
    if api_key is None:
        raise ValueError(
            "FRED_API_KEY가 설정되어 있지 않습니다. "
            "환경변수 또는 .env 파일에 FRED_API_KEY를 설정하세요."
        )
    return Fred(api_key=api_key)


def get_existing_fred_range(series_id: str) -> Optional[pd.DatetimeIndex]:
    file_path = FRED_DIR / f"{series_id}.parquet"
    if not file_path.exists():
        return None
    try:
        s = pd.read_parquet(file_path)["value"]
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return s.index
    except Exception:
        return None


def download_fred_series(
    series_ids: List[str],
    start: str,
    end: str,
    force: bool = False,
    api_key: Optional[str] = None,
) -> Dict[str, pd.Series]:
    """
    FRED에서 시계열들을 다운로드하고 로컬 캐시(parquet)에 저장.
    각 series_id별로 parquet 파일 하나 생성.
    """
    fred = get_fred_client(api_key)
    start_dt = parse_date(start)
    end_dt = parse_date(end)

    result = {}
    for sid in series_ids:
        file_path = FRED_DIR / f"{sid}.parquet"
        existing_range = get_existing_fred_range(sid)
        existing_s = None
        if existing_range is not None and not force:
            df = pd.read_parquet(file_path)
            s = df["value"]
            s.index = pd.to_datetime(s.index).tz_localize(None)
            existing_s = s

        download_start = start_dt
        if existing_range is not None and not force:
            last_date = existing_range.max()
            if last_date >= end_dt:
                print(f"[FRED] {sid}: already up-to-date ({file_path})")
                result[sid] = existing_s.loc[
                    (existing_s.index >= start_dt) & (existing_s.index <= end_dt)
                ]
                continue
            else:
                download_start = last_date + pd.Timedelta(days=1)

        print(f"[FRED] Download {sid}: {download_start.date()} → {end_dt.date()}")
        # FRED는 보통 월/주/일 빈도, 데이터 없음 날짜는 NaN
        new_s = fred.get_series(
            sid,
            observation_start=download_start.strftime("%Y-%m-%d"),
            observation_end=end_dt.strftime("%Y-%m-%d"),
        )
        new_s.index = pd.to_datetime(new_s.index).tz_localize(None)
        new_s = new_s.sort_index()

        if existing_s is not None and not force:
            combined = pd.concat([existing_s, new_s])
            combined = combined[~combined.index.duplicated(keep="first")]
        else:
            combined = new_s

        combined = combined.sort_index()
        df_save = pd.DataFrame({"value": combined})
        df_save.to_parquet(file_path)
        print(f"[FRED] Saved {sid} to {file_path} (rows={len(combined)})")

        result[sid] = combined.loc[
            (combined.index >= start_dt) & (combined.index <= end_dt)
        ]

    return result


# -------------------------------------------------------------------
# 메인: CLI 인터페이스
# -------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="v1 데이터 수집기 (Yahoo + FRED)")
    p.add_argument(
        "--tickers-file",
        type=str,
        required=True,
        help="라인별로 ticker가 적힌 텍스트 파일 경로",
    )
    p.add_argument(
        "--start",
        type=str,
        required=True,
        help="시작 날짜 (YYYY-MM-DD)",
    )
    p.add_argument(
        "--end",
        type=str,
        required=True,
        help="종료 날짜 (YYYY-MM-DD)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="기존 캐시 무시하고 전체 구간 다시 다운로드",
    )
    p.add_argument(
        "--fred-series",
        nargs="*",
        default=[],
        help="FRED series ID 리스트 (공백으로 구분). 예: DGS10 T10Y2Y VIXCLS",
    )
    p.add_argument(
        "--fred-api-key",
        type=str,
        default=None,
        help="FRED API 키 (옵션, 미지정 시 환경변수/ .env 에서 읽음)",
    )
    return p


def main():
    args = build_arg_parser().parse_args()

    tickers_file = Path(args.tickers_file)
    if not tickers_file.exists():
        raise FileNotFoundError(f"Tickers file not found: {tickers_file}")

    tickers = load_tickers_from_file(tickers_file)
    print(f"Loaded {len(tickers)} tickers from {tickers_file}")

    # 1) Yahoo Finance 다운로드
    yahoo_data = download_yahoo_prices(
        tickers=tickers,
        start=args.start,
        end=args.end,
        force=args.force,
        interval="1d",
    )
    print(f"[Yahoo] Download complete for {len(yahoo_data)} tickers.")

    # 2) FRED 다운로드 (선택)
    fred_data = {}
    if args.fred_series:
        fred_data = download_fred_series(
            series_ids=args.fred_series,
            start=args.start,
            end=args.end,
            force=args.force,
            api_key=args.fred_api_key,
        )
        print(f"[FRED] Download complete for {len(fred_data)} series.")

    print("\n=== v1 데이터 수집 완료 ===")
    print(f"Yahoo data saved under: {YAHOO_DIR}")
    print(f"FRED data saved under:  {FRED_DIR}")
    print("이제 feature tensor builder / no-look-ahead aligner 단계로 넘어가면 됩니다.")


if __name__ == "__main__":
    main()
