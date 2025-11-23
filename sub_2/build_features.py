#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from fredapi import Fred

# --------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------
# Helper: Read FRED API key from .env
# --------------------------------------------------------------------
def load_fred_api_key():
    env_path = os.path.join(os.getcwd(), ".env")
    if not os.path.exists(env_path):
        raise FileNotFoundError(".env file not found.")

    with open(env_path, "r") as f:
        for line in f:
            if "FRED_API_KEY" in line:
                return line.strip().split("=")[1]
    raise ValueError("FRED_API_KEY not found in .env")


FRED_API_KEY = load_fred_api_key()
fred = Fred(api_key=FRED_API_KEY)


# --------------------------------------------------------------------
# Macro series list
# --------------------------------------------------------------------
FRED_SERIES = [
    ("DFF", "fed_funds_rate"),
    ("T10Y3M", "term_spread_10y_3m"),
    ("BAA10YM", "credit_spread_baa10y"),
    ("VIXCLS", "vix_index"),
    ("CPIAUCSL", "cpi_index"),
    ("INDPRO", "industrial_production_idx"),
    ("DTWEXBGS", "usd_broad_index"),
]


# --------------------------------------------------------------------
# Fetch FRED series (existing signature)
# --------------------------------------------------------------------
def fetch_fred_series(series_id, start, end, col_name):
    """
    Existing pipeline version: fetch_fred_series(series_id, start, end, col_name)
    """
    try:
        s = fred.get_series(series_id)
        s = s.to_frame(col_name)
        s.index = pd.to_datetime(s.index)
        s = s.loc[(s.index >= start) & (s.index <= end)]
        return s
    except Exception as e:
        logger.error(f"Failed to fetch FRED series {series_id}: {e}")
        return pd.DataFrame(columns=[col_name])


# --------------------------------------------------------------------
# Technical Indicators
# --------------------------------------------------------------------
def compute_technicals(df: pd.DataFrame) -> pd.DataFrame:
    """
    df index = Date, columns = ['Open','High','Low','Close','Volume']
    """
    out = pd.DataFrame(index=df.index)

    # 1) log-return
    out["ret_1d"] = np.log(df["Close"]).diff()

    # 2) RSI 14
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / roll_down
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # 3~5) SMA
    out["sma_20"] = df["Close"].rolling(20).mean()
    out["sma_50"] = df["Close"].rolling(50).mean()
    out["sma_200"] = df["Close"].rolling(200).mean()

    # 6~8) MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    out["macd"] = macd
    out["macd_signal"] = signal
    out["macd_hist"] = macd - signal

    # 9~11) Bollinger (20)
    ma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    out["bb_high_20"] = ma20 + 2 * std20
    out["bb_low_20"] = ma20 - 2 * std20
    out["bb_width_20"] = out["bb_high_20"] - out["bb_low_20"]

    # 12) ATR 14
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    out["atr_14"] = tr.rolling(14).mean()

    # 13~14) Stochastic
    low14 = df["Low"].rolling(14).min()
    high14 = df["High"].rolling(14).max()
    k = 100 * (df["Close"] - low14) / (high14 - low14)
    d = k.rolling(3).mean()
    out["stoch_k_14_3"] = k
    out["stoch_d_14_3"] = d

    # 15) ROC 10
    out["roc_10"] = df["Close"].pct_change(10)

    # 16~17) Realized vol
    out["realized_vol_20"] = np.log(df["Close"]).diff().rolling(20).std()
    out["realized_vol_60"] = np.log(df["Close"]).diff().rolling(60).std()

    # 18) Volume ROC
    out["volume_roc_20"] = df["Volume"].pct_change(20)

    return out


# --------------------------------------------------------------------
# Macro panel builder
# --------------------------------------------------------------------
def build_macro_panel(start, end):
    logger.info("Fetching / building macro variables...")

    macro_raw = {}
    for series_id, col_name in FRED_SERIES:
        logger.info(f"Fetching FRED series: {series_id} as {col_name}")
        s = fetch_fred_series(series_id, start, end, col_name)
        macro_raw[col_name] = s[col_name]

    macro = pd.DataFrame(macro_raw).sort_index()

    # CPI YoY
    if "cpi_index" in macro.columns:
        macro["inflation_cpi_yoy"] = macro["cpi_index"].pct_change(12) * 100

    # Industrial production YoY
    if "industrial_production_idx" in macro.columns:
        macro["industrial_production_yoy"] = (
            macro["industrial_production_idx"].pct_change(12) * 100
        )

    macro = macro.ffill()

    macro_cols = [
        "vix_index",
        "term_spread_10y_3m",
        "credit_spread_baa10y",
        "inflation_cpi_yoy",
        "industrial_production_yoy",
        "usd_broad_index",
        "fed_funds_rate",
    ]

    macro = macro[macro_cols]
    logger.info(f"Macro DataFrame shape: {macro.shape}")

    return macro


# --------------------------------------------------------------------
# Main Build Process
# --------------------------------------------------------------------
def main():
    START = "2005-01-01"
    END   = "2025-12-31"

    ohlcv_path = "data/equity_ohlcv_2005_2025.parquet"
    feature_out_path = "data/feature_panel.parquet"
    macro_out_path = "data/macro_panel.parquet"

    logger.info(f"Loading OHLCV data from {ohlcv_path}")
    df = pd.read_parquet(ohlcv_path)
    df.index = pd.MultiIndex.from_frame(df.index.to_frame())
    df.index.set_names(["Date", "Ticker"], inplace=True)

    # --------------------------------------------------------------
    # Compute technicals per ticker
    # --------------------------------------------------------------
    logger.info("Starting feature engineering for panel.")
    logger.info("Computing features by ticker (this may take a while)...")

    panel_list = []

    for ticker, g in df.groupby("Ticker"):
        g0 = g.droplevel("Ticker")
        feats = compute_technicals(g0)
        feats["Open"] = g0["Open"]
        feats["High"] = g0["High"]
        feats["Low"] = g0["Low"]
        feats["Close"] = g0["Close"]
        feats["Volume"] = g0["Volume"]
        feats["Ticker"] = ticker
        panel_list.append(feats)

    feature_panel = pd.concat(panel_list)
    feature_panel = feature_panel.reset_index().set_index(["Date", "Ticker"])

    # Filter tickers that have at least 252 days
    logger.info("Filtering tickers with at least 252 observations.")
    counts = feature_panel.groupby("Ticker")["Close"].count()
    keep = counts[counts >= 252].index
    feature_panel = feature_panel.loc[feature_panel.index.get_level_values("Ticker").isin(keep)]

    logger.info(f"Remaining tickers after history filter: {len(keep)}")
    logger.info(f"Feature panel shape (with technicals): {feature_panel.shape}")

    # --------------------------------------------------------------
    # Fetch macro panel
    # --------------------------------------------------------------
    macro_panel = build_macro_panel(START, END)
    macro_panel.to_parquet(macro_out_path)
    logger.info(f"Saved macro panel to {macro_out_path}")

    # --------------------------------------------------------------
    # Save feature panel
    # --------------------------------------------------------------
    feature_panel.to_parquet(feature_out_path)
    logger.info(f"Saving feature panel to {feature_out_path}")

    logger.info("All done.")


if __name__ == "__main__":
    main()
