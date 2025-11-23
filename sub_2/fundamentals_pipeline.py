# fundamentals_pipeline.py
"""
Fundamental feature pipeline.

- Input:
    - data/sp500_tickers.csv           : S&P 500 tickers list
    - data/equity_ohlcv_2005_2025.parquet : OHLCV panel (for trading-day index)

- Output:
    - data/fundamentals_raw.parquet    : raw fundamentals at reporting dates
    - data/fundamentals_panel.parquet  : daily (Date, Ticker) panel, forward-filled

Features targeted (per ticker):
    - pe_ttm
    - pb_mrq
    - roe
    - roa
    - debt_to_equity
    - dividend_yield
    - payout_ratio
    - asset_turnover
    - fcf_yield
    (정확한 정의는 사용 가능한 항목에 따라 근사)

Note:
    yfinance 기반 무료 fundamentals는 커버리지/히스토리 한계가 있으므로,
    논문에서는 "공개 API 기반 근사값"으로 명시하고 사용해야 한다.
"""

import os
import logging
from typing import Dict, List

import numpy as np
import pandas as pd
import yfinance as yf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

TICKER_CSV = os.path.join(DATA_DIR, "sp500_tickers.csv")
OHLCV_PATH = os.path.join(DATA_DIR, "equity_ohlcv_2005_2025.parquet")
RAW_FUND_PATH = os.path.join(DATA_DIR, "fundamentals_raw.parquet")
PANEL_FUND_PATH = os.path.join(DATA_DIR, "fundamentals_panel.parquet")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------
# Helpers
# --------------------------------------------------


def load_tickers() -> List[str]:
    if not os.path.exists(TICKER_CSV):
        raise FileNotFoundError(f"Ticker CSV not found: {TICKER_CSV}")
    df = pd.read_csv(TICKER_CSV)
    # 컬럼 이름이 'Symbol' 또는 'Ticker'일 가능성 있음
    for col in ["Symbol", "Ticker", "symbol", "ticker"]:
        if col in df.columns:
            tickers = sorted(df[col].dropna().unique().tolist())
            logger.info(f"Loaded {len(tickers)} tickers from {TICKER_CSV}")
            return tickers
    raise ValueError(
        f"Could not find ticker column in {TICKER_CSV}. "
        "Expected one of ['Symbol','Ticker','symbol','ticker']."
    )


def load_trading_calendar() -> pd.DatetimeIndex:
    if not os.path.exists(OHLCV_PATH):
        raise FileNotFoundError(f"OHLCV parquet not found: {OHLCV_PATH}")
    df = pd.read_parquet(OHLCV_PATH)
    if isinstance(df.index, pd.MultiIndex):
        dates = df.index.get_level_values("Date").unique()
    else:
        if "Date" not in df.columns:
            raise ValueError("OHLCV must contain 'Date' column.")
        dates = pd.to_datetime(df["Date"]).unique()
    dates = pd.DatetimeIndex(sorted(dates))
    logger.info(f"Trading calendar loaded: {len(dates)} dates.")
    return dates


def safe_get(df: pd.DataFrame, row: str, col: str):
    try:
        return df.loc[row, col]
    except Exception:
        return np.nan


def compute_fundamentals_for_ticker(ticker: str) -> pd.DataFrame:
    """
    Fetch fundamentals for a single ticker using yfinance.

    This is a best-effort approximation:
      - Uses annual statements if available.
      - Some ratios derived from basic line items.
    """
    logger.info(f"[{ticker}] Fetching fundamentals via yfinance...")
    tk = yf.Ticker(ticker)

    try:
        bs = tk.get_balancesheet(freq="yearly")
    except Exception:
        bs = None
    try:
        is_ = tk.get_income_stmt(freq="yearly")
    except Exception:
        is_ = None
    try:
        cf = tk.get_cashflow(freq="yearly")
    except Exception:
        cf = None
    try:
        info = tk.get_info()
    except Exception:
        info = {}

    if (bs is None or bs.empty) and (is_ is None or is_.empty):
        logger.warning(f"[{ticker}] No balance sheet or income statement. Skipping.")
        return pd.DataFrame()

    # transpose: columns are dates
    if bs is not None and not bs.empty:
        bs = bs.T  # index: period end date
    if is_ is not None and not is_.empty:
        is_ = is_.T
    if cf is not None and not cf.empty:
        cf = cf.T

    # unify index name
    for df in [bs, is_, cf]:
        if df is not None and not df.empty:
            df.index = pd.to_datetime(df.index)
            df.index.name = "Date"

    # -------------------
    # compute ratios
    # -------------------
    rows = []

    # market-cap-based items from info (mostly current, not historical)
    market_cap = info.get("marketCap", np.nan)
    trailing_pe = info.get("trailingPE", np.nan)
    price_to_book = info.get("priceToBook", np.nan)
    dividend_yield = info.get("dividendYield", np.nan)
    payout_ratio = info.get("payoutRatio", np.nan)

    # We'll attach these static ratios as "most recent" to the last financial date
    static_ratios = {
        "pe_ttm": trailing_pe,
        "pb_mrq": price_to_book,
        "dividend_yield": dividend_yield,
        "payout_ratio": payout_ratio,
    }

    if bs is None:
        bs = pd.DataFrame()
    if is_ is None:
        is_ = pd.DataFrame()
    if cf is None:
        cf = pd.DataFrame()

    all_dates = sorted(set(bs.index) | set(is_.index) | set(cf.index))
    if not all_dates:
        logger.warning(f"[{ticker}] No dated fundamentals. Skipping.")
        return pd.DataFrame()

    for dt in all_dates:
        bs_row = bs.loc[dt] if dt in bs.index else pd.Series(dtype=float)
        is_row = is_.loc[dt] if dt in is_.index else pd.Series(dtype=float)
        cf_row = cf.loc[dt] if dt in cf.index else pd.Series(dtype=float)

        # 항목 이름은 yfinance 버전에 따라 조금 다를 수 있으므로, 여러 후보를 시도
        total_assets = (
            safe_get(bs.T, "Total Assets", dt)
            or safe_get(bs.T, "TotalAssets", dt)
        )
        total_equity = (
            safe_get(bs.T, "Total Stockholder Equity", dt)
            or safe_get(bs.T, "Total Equity Gross Minority Interest", dt)
        )
        total_liab = (
            safe_get(bs.T, "Total Liabilities Net Minority Interest", dt)
            or safe_get(bs.T, "Total Liab", dt)
        )
        total_debt = (
            safe_get(bs.T, "Total Debt", dt)
            or safe_get(bs.T, "Short Long Term Debt", dt)
        )
        total_rev = (
            safe_get(is_.T, "Total Revenue", dt)
            or safe_get(is_.T, "Revenue", dt)
        )
        net_income = (
            safe_get(is_.T, "Net Income", dt)
            or safe_get(is_.T, "NetIncome", dt)
        )
        fcf = (
            safe_get(cf.T, "Free Cash Flow", dt)
            or safe_get(cf.T, "FreeCashFlow", dt)
        )

        # ratios
        roe = (
            net_income / (total_equity + 1e-12)
            if not (pd.isna(net_income) or pd.isna(total_equity))
            else np.nan
        )
        roa = (
            net_income / (total_assets + 1e-12)
            if not (pd.isna(net_income) or pd.isna(total_assets))
            else np.nan
        )
        debt_to_equity = (
            total_debt / (total_equity + 1e-12)
            if not (pd.isna(total_debt) or pd.isna(total_equity))
            else np.nan
        )
        asset_turnover = (
            total_rev / (total_assets + 1e-12)
            if not (pd.isna(total_rev) or pd.isna(total_assets))
            else np.nan
        )
        fcf_yield = (
            fcf / (market_cap + 1e-12)
            if not (pd.isna(fcf) or pd.isna(market_cap))
            else np.nan
        )

        row: Dict = {
            "Ticker": ticker,
            "Date": dt,
            "total_assets": total_assets,
            "total_equity": total_equity,
            "total_debt": total_debt,
            "total_revenue": total_rev,
            "net_income": net_income,
            "free_cash_flow": fcf,
            "roe": roe,
            "roa": roa,
            "debt_to_equity": debt_to_equity,
            "asset_turnover": asset_turnover,
            "fcf_yield": fcf_yield,
        }
        rows.append(row)

    raw_df = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)

    # 가장 최근 기간에 static ratios 부여 (이건 엄밀히 말해 시계열이 아니라 최근 값 근사)
    if not raw_df.empty:
        latest_idx = raw_df["Date"].idxmax()
        for k, v in static_ratios.items():
            raw_df.loc[latest_idx, k] = v

    return raw_df


# --------------------------------------------------
# Build full fundamentals
# --------------------------------------------------


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    tickers = load_tickers()
    dates = load_trading_calendar()

    all_raw: List[pd.DataFrame] = []

    for i, tk in enumerate(tickers, start=1):
        logger.info(f"[{i}/{len(tickers)}] Processing {tk}")
        try:
            raw = compute_fundamentals_for_ticker(tk)
        except Exception as e:
            logger.exception(f"Error while fetching fundamentals for {tk}: {e}")
            continue
        if raw is not None and not raw.empty:
            all_raw.append(raw)

    if not all_raw:
        logger.warning("No fundamentals collected for any ticker.")
        return

    raw_all = pd.concat(all_raw, axis=0, ignore_index=True)
    raw_all["Date"] = pd.to_datetime(raw_all["Date"])
    raw_all = raw_all.sort_values(["Ticker", "Date"])
    logger.info(f"Raw fundamentals shape: {raw_all.shape}")

    # Save raw (reporting-date-level) fundamentals
    logger.info(f"Saving raw fundamentals to {RAW_FUND_PATH}")
    raw_all.to_parquet(RAW_FUND_PATH)

    # --------------------------------------------------
    # Build (Date, Ticker) daily panel via forward-fill
    # --------------------------------------------------
    logger.info("Building daily fundamentals panel via forward-fill...")

    panel_list: List[pd.DataFrame] = []

    for tk in raw_all["Ticker"].unique():
        sub = raw_all[raw_all["Ticker"] == tk].copy()
        sub = sub.set_index("Date").sort_index()

        # reindex to full calendar, then forward-fill
        tmp = sub.reindex(dates).ffill()

        # attach ticker and reset index
        tmp["Ticker"] = tk
        tmp = tmp.reset_index().rename(columns={"index": "Date"})
        panel_list.append(tmp)

    fund_panel = pd.concat(panel_list, axis=0, ignore_index=True)
    fund_panel = fund_panel.set_index(["Date", "Ticker"]).sort_index()
    logger.info(f"Fundamentals daily panel shape: {fund_panel.shape}")

    logger.info(f"Saving fundamentals panel to {PANEL_FUND_PATH}")
    fund_panel.to_parquet(PANEL_FUND_PATH)

    logger.info("Fundamentals pipeline completed.")


if __name__ == "__main__":
    main()
