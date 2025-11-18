#!/usr/bin/env python
# build_features_v1.py
"""
v1 Feature Tensor Builder + No-Look-Ahead Aligner

- Yahoo Finance parquet (per-ticker) + FRED parquet를 읽어서
  (date, ticker) 패널로 정리
- 가격 기반 / 간단 테크니컬 / 매크로 feature 생성
- no-look-ahead 를 지키는 forward 1d return 라벨 생성
- 최종 결과를 panel_features.parquet 로 저장

추가:
- RollingPanelDataset 클래스로 240×N×F 윈도우를 on-the-fly 생성 가능
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# 1. 기본 유틸
# -----------------------------

def _log(msg: str):
    print(f"[build_features_v1] {msg}")


# -----------------------------
# 2. 데이터 로더
# -----------------------------

def load_yahoo_panel(yahoo_dir: Path) -> pd.DataFrame:
    """
    yahoo_dir 아래 *.parquet을 읽어 (date, ticker) MultiIndex 패널로 합침.

    기대 형식 (티커별 파일, 예: AAPL.parquet):
    - 컬럼에 Date / date, Open / open, High / high, Low / low, Close / close,
      Adj Close / adj_close, Volume / volume 등이 있을 것이라 가정.
    """
    files = list(yahoo_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {yahoo_dir}")

    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        # 컬럼명 lower-case + 간단 통일
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        # date 컬럼 / index 정리
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        else:
            # index가 날짜인 경우
            df = df.reset_index()
            if "index" in df.columns:
                df.rename(columns={"index": "date"}, inplace=True)
            df["date"] = pd.to_datetime(df["date"])

        # 가격 컬럼 명 통일
        rename_map = {}
        if "adj close" in df.columns:
            rename_map["adj close"] = "adj_close"
        if "adj_close" in df.columns:
            # already fine
            pass
        if "open" not in df.columns and "open" in df.columns:
            rename_map["open"] = "open"
        if "close" not in df.columns and "close" in df.columns:
            rename_map["close"] = "close"
        if "high" not in df.columns and "high" in df.columns:
            rename_map["high"] = "high"
        if "low" not in df.columns and "low" in df.columns:
            rename_map["low"] = "low"
        if "volume" not in df.columns and "volume" in df.columns:
            rename_map["volume"] = "volume"

        df = df.rename(columns=rename_map)

        # 티커 이름은 파일명(stem)에서 가져온다
        ticker = f.stem.upper()
        df["ticker"] = ticker

        # 필요한 컬럼만
        cols_keep = []
        for c in ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]:
            if c in df.columns:
                cols_keep.append(c)
        df = df[cols_keep]

        dfs.append(df)

    all_df = pd.concat(dfs, axis=0, ignore_index=True)
    all_df = all_df.drop_duplicates(subset=["date", "ticker"])
    all_df = all_df.sort_values(["ticker", "date"])

    # MultiIndex 패널로 변환
    all_df.set_index(["date", "ticker"], inplace=True)
    all_df = all_df.sort_index()

    _log(f"Loaded yahoo panel: {all_df.shape[0]} rows, {all_df.index.get_level_values('ticker').nunique()} tickers")
    return all_df


def load_fred_panel(fred_dir: Path) -> pd.DataFrame:
    """
    fred_dir 아래 parquet을 읽어서 date index를 가진 DataFrame으로 합침.
    각 컬럼이 서로 다른 시계열이 되게 만든다.

    기대 형식:
    - 1) fred.parquet 하나에 여러 컬럼이 있을 수도 있고
    - 2) series별로 여러 parquet 파일이 있을 수도 있음:
         columns: ["date", "value"] or ["date", "DGS10"] 형태

    여기서는 가능한 한 자동으로 합쳐서:
        index = date, columns = series_name
    로 만든다.
    """
    files = list(fred_dir.glob("*.parquet"))
    if not files:
        _log(f"No FRED parquet found under {fred_dir}, returning empty DataFrame.")
        return pd.DataFrame()

    series_dfs = []

    for f in files:
        df = pd.read_parquet(f)
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        else:
            df = df.reset_index()
            df.rename(columns={"index": "date"}, inplace=True)
            df["date"] = pd.to_datetime(df["date"])

        # "value" 하나만 있고 파일명이 시리즈인 경우
        if "value" in df.columns and df.shape[1] == 2:
            series_name = f.stem  # 예: DGS10.parquet -> "DGS10"
            df = df.rename(columns={"value": series_name})
            df = df[["date", series_name]]
        else:
            # 이미 여러 컬럼일 수 있음
            pass

        df = df.set_index("date").sort_index()
        series_dfs.append(df)

    fred_df = pd.concat(series_dfs, axis=1)
    fred_df = fred_df.sort_index()
    fred_df = fred_df.loc[~fred_df.index.duplicated(keep="first")]

    _log(f"Loaded FRED data: {fred_df.shape[0]} days, {fred_df.shape[1]} series")
    return fred_df


# -----------------------------
# 3. 테크니컬 인디케이터 (v1)
# -----------------------------

def compute_technical_features(price_panel: pd.DataFrame) -> pd.DataFrame:
    """
    price_panel: index = (date, ticker)
    columns: ['open','high','low','close','adj_close','volume'] (subset 포함 가능)

    v1에서 구현할 feature 예:
    - log_return (1d)
    - rolling_vol_20d
    - rolling_vol_60d
    - rsi_14
    - macd (12,26,9)
    - bb_upper_20, bb_lower_20 (볼린저 밴드)
    - atr_14
    - obv
    """
    df = price_panel.copy()

    if "adj_close" in df.columns:
        px = df["adj_close"]
    else:
        px = df["close"]

    # groupby ticker
    def by_ticker(func):
        return df.groupby(level="ticker", group_keys=False).apply(func)

    # log return
    df["log_ret_1d"] = by_ticker(lambda x: np.log(x["adj_close"].fillna(method="ffill")).diff()
                                 if "adj_close" in x.columns
                                 else np.log(x["close"].fillna(method="ffill")).diff())

    # realized vol
    df["vol_20d"] = by_ticker(lambda x: x["log_ret_1d"].rolling(20, min_periods=5).std() * np.sqrt(252))
    df["vol_60d"] = by_ticker(lambda x: x["log_ret_1d"].rolling(60, min_periods=10).std() * np.sqrt(252))

    # RSI 14
    def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window, min_periods=window).mean()
        avg_loss = loss.rolling(window, min_periods=window).mean()
        rs = avg_gain / (avg_loss + 1e-12)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    df["rsi_14"] = by_ticker(lambda x: _rsi(x["adj_close"].fillna(method="ffill")
                                            if "adj_close" in x.columns else x["close"].fillna(method="ffill"), 14))

    # MACD (12,26) + signal(9)
    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    def _macd(x: pd.Series) -> pd.DataFrame:
        ema12 = _ema(x, 12)
        ema26 = _ema(x, 26)
        macd_line = ema12 - ema26
        signal = _ema(macd_line, 9)
        hist = macd_line - signal
        return pd.DataFrame({"macd": macd_line, "macd_signal": signal, "macd_hist": hist}, index=x.index)

    macd_df = by_ticker(lambda x: _macd(x["adj_close"].fillna(method="ffill")
                                        if "adj_close" in x.columns else x["close"].fillna(method="ffill")))
    df = df.join(macd_df)

    # Bollinger Bands (20d, 2σ)
    def _bb(x: pd.Series, window: int = 20, k: float = 2.0) -> pd.DataFrame:
        ma = x.rolling(window, min_periods=window).mean()
        sd = x.rolling(window, min_periods=window).std()
        upper = ma + k * sd
        lower = ma - k * sd
        return pd.DataFrame({"bb_mid_20": ma, "bb_upper_20": upper, "bb_lower_20": lower}, index=x.index)

    bb_df = by_ticker(lambda x: _bb(x["adj_close"].fillna(method="ffill")
                                    if "adj_close" in x.columns else x["close"].fillna(method="ffill")))
    df = df.join(bb_df)

    # ATR 14
    def _atr(x: pd.DataFrame, window: int = 14) -> pd.Series:
        high = x["high"]
        low = x["low"]
        if "adj_close" in x.columns:
            close_prev = x["adj_close"].shift(1)
        else:
            close_prev = x["close"].shift(1)

        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window, min_periods=window).mean()
        return atr

    if {"high", "low", "close"}.issubset(df.columns):
        df["atr_14"] = by_ticker(lambda x: _atr(x, 14))

    # OBV
    def _obv(x: pd.DataFrame) -> pd.Series:
        if "adj_close" in x.columns:
            px = x["adj_close"]
        else:
            px = x["close"]
        vol = x["volume"].fillna(0)
        sign = np.sign(px.diff().fillna(0))
        return (sign * vol).cumsum()

    if {"close", "volume"}.issubset(df.columns) or {"adj_close", "volume"}.issubset(df.columns):
        df["obv"] = by_ticker(_obv)

    _log("Computed technical features.")
    return df


# -----------------------------
# 4. 매크로 결합 + 라벨 생성
# -----------------------------

def merge_macro_features(
    panel: pd.DataFrame,
    fred_df: pd.DataFrame
) -> pd.DataFrame:
    """
    price/tech 패널에 FRED 매크로 시계열을 date 기준으로 merge.
    매크로는 모든 ticker에 동일하게 붙음.
    """
    if fred_df is None or fred_df.empty:
        _log("No FRED data to merge. Skipping macro features.")
        return panel

    fred = fred_df.copy()
    fred.index = pd.to_datetime(fred.index)
    fred = fred.sort_index()
    # forward-fill: macro 데이터의 missing은 과거 값 유지
    fred = fred.ffill()

    # panel index: (date, ticker)
    df = panel.reset_index()
    df["date"] = pd.to_datetime(df["date"])

    fred_reset = fred.reset_index().rename(columns={"index": "date"})
    merged = df.merge(fred_reset, on="date", how="left")

    # 매크로도 forward-fill (ticker별이 아니라 전체 시간축 기준)
    merged = merged.sort_values(["ticker", "date"])
    macro_cols = [c for c in fred.columns]
    merged[macro_cols] = merged.groupby("ticker")[macro_cols].ffill()

    merged = merged.set_index(["date", "ticker"]).sort_index()
    _log("Merged macro features into panel.")
    return merged


def add_forward_returns(panel: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    horizon(기본 1일) forward return을 no-look-ahead로 라벨로 추가.

    fwd_ret_1d(t) = (Px_{t+1} / Px_t - 1)
    """
    df = panel.copy()

    def _fwd_ret(x: pd.DataFrame) -> pd.Series:
        if "adj_close" in x.columns:
            px = x["adj_close"]
        else:
            px = x["close"]
        return px.shift(-horizon) / px - 1.0

    df[f"fwd_ret_{horizon}d"] = df.groupby(level="ticker", group_keys=False).apply(_fwd_ret)

    # horizon 이후에 라벨이 없는 마지막 구간은 drop
    # (미래가 없으므로 학습에 쓸 수 없음)
    df = df.groupby(level="ticker", group_keys=False).apply(lambda x: x.iloc[:-horizon])

    _log(f"Added forward {horizon}d returns as label (no-look-ahead).")
    return df


# -----------------------------
# 5. 최종 패널 저장 + (선택) Dataset
# -----------------------------

@dataclass
class FeatureBuildConfig:
    yahoo_dir: Path
    fred_dir: Optional[Path]
    out_path: Path
    forward_horizon: int = 1


def build_feature_panel(cfg: FeatureBuildConfig) -> pd.DataFrame:
    yahoo_panel = load_yahoo_panel(cfg.yahoo_dir)
    fred_df = load_fred_panel(cfg.fred_dir) if cfg.fred_dir is not None else pd.DataFrame()

    panel = compute_technical_features(yahoo_panel)
    panel = merge_macro_features(panel, fred_df)
    panel = add_forward_returns(panel, horizon=cfg.forward_horizon)

    # 정리: 너무 앞부분 초기 구간(테크니컬 rolling window)에서 NaN이 많은 행은 제거
    # NaN 비율 기준으로 필터링
    non_label_cols = [c for c in panel.columns if not c.startswith("fwd_ret_")]
    panel = panel.dropna(subset=non_label_cols, how="any")  # v1에서는 과감하게 any drop

    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(cfg.out_path)

    _log(f"Saved feature panel to {cfg.out_path} with shape {panel.shape}")
    return panel


# (선택) 240×N×F 윈도우를 만드는 Dataset-like 클래스
class RollingPanelDataset:
    """
    Panel DataFrame (date, ticker index)를 받아서
    (lookback, N_tickers, F_features) 윈도우를 생성하는 헬퍼.

    실제 학습 코드에서는 이걸 PyTorch Dataset으로 래핑하면 된다.
    """

    def __init__(
        self,
        panel: pd.DataFrame,
        lookback: int = 240,
        feature_cols: Optional[Sequence[str]] = None,
        label_col: str = "fwd_ret_1d",
        tickers: Optional[Sequence[str]] = None,
    ):
        """
        panel: index = (date, ticker)
        feature_cols: 사용할 feature 컬럼 리스트. None이면 label_col 제외한 모든 numeric 컬럼 사용.
        """
        self.lookback = lookback
        self.label_col = label_col

        # 패널을 (date, ticker) 기준으로 pivot -> Panel 형태
        df = panel.copy()
        df = df.sort_index()

        # 사용 tickers 고정
        if tickers is None:
            self.tickers = sorted(df.index.get_level_values("ticker").unique())
        else:
            self.tickers = list(tickers)

        # feature/label 분리
        all_cols = df.columns.tolist()
        if feature_cols is None:
            feature_cols = [c for c in all_cols if c != label_col]
        self.feature_cols = list(feature_cols)

        # pivot: date x (ticker, feature)
        # -> 나중에 np reshape으로 (date, ticker, feature) 만들 예정
        df_feat = df[self.feature_cols]
        df_label = df[[label_col]]

        feat_panel = df_feat.unstack("ticker")  # index=date, columns=(feature, ticker)
        label_panel = df_label.unstack("ticker")  # index=date, columns=(label, ticker)

        # tickers & feature 순서를 정렬/고정
        feat_panel = feat_panel.reindex(columns=pd.MultiIndex.from_product([self.feature_cols, self.tickers]))
        label_panel = label_panel.reindex(columns=pd.MultiIndex.from_product([[label_col], self.tickers]))

        # 공통 date index
        common_dates = feat_panel.index.intersection(label_panel.index)
        self.dates = common_dates

        self.feat_panel = feat_panel.loc[common_dates]
        self.label_panel = label_panel.loc[common_dates]

        # NaN 제거 (v1 단순 버전: NaN 포함된 row를 통째로 drop)
        # 실제로는 더 정교하게 할 수 있음
        mask_valid = (~self.feat_panel.isna().any(axis=1)) & (~self.label_panel.isna().any(axis=1))
        self.feat_panel = self.feat_panel.loc[mask_valid]
        self.label_panel = self.label_panel.loc[mask_valid]
        self.dates = self.feat_panel.index

        # lookback 만드는 anchor index 범위
        self.anchor_indices = np.arange(lookback - 1, len(self.dates) - 1)  # 마지막은 label horizon 고려

        _log(
            f"RollingPanelDataset: {len(self.dates)} dates, "
            f"{len(self.tickers)} tickers, {len(self.feature_cols)} features, "
            f"{len(self)} windows with lookback={lookback}"
        )

    def __len__(self):
        return len(self.anchor_indices)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, pd.Timestamp]:
        """
        반환:
        - X: (lookback, N_tickers, F_features)
        - y: (N_tickers,)  (anchor_date의 label_col)
        - anchor_date: 해당 샘플의 기준 날짜
        """
        anchor_pos = self.anchor_indices[idx]
        anchor_date = self.dates[anchor_pos]

        date_slice = self.dates[anchor_pos - self.lookback + 1 : anchor_pos + 1]

        # feature: (T_dates, feature*ticker MultiIndex)
        feat_slice = self.feat_panel.loc[date_slice]
        # reshape: (T_dates, F, N_tickers) -> (T_dates, N_tickers, F)
        F = len(self.feature_cols)
        N = len(self.tickers)
        X = feat_slice.values.reshape(len(date_slice), F, N)
        X = np.transpose(X, (0, 2, 1))

        # label: anchor_date에서의 forward return
        y_slice = self.label_panel.loc[anchor_date]
        y = y_slice.values.reshape(1, N)[0]

        return X.astype(np.float32), y.astype(np.float32), anchor_date


# -----------------------------
# 6. CLI
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="v1 Feature Tensor Builder (Yahoo + FRED)")
    p.add_argument("--yahoo-dir", type=str, required=True, help="Yahoo parquet directory")
    p.add_argument("--fred-dir", type=str, required=False, help="FRED parquet directory (optional)")
    p.add_argument("--out-path", type=str, required=True, help="Output parquet path for feature panel")
    p.add_argument("--horizon", type=int, default=1, help="Forward return horizon (days)")
    return p.parse_args()


def main():
    args = parse_args()
    yahoo_dir = Path(args.yahoo_dir)
    fred_dir = Path(args.fred_dir) if args.fred_dir is not None else None
    out_path = Path(args.out_path)

    cfg = FeatureBuildConfig(
        yahoo_dir=yahoo_dir,
        fred_dir=fred_dir,
        out_path=out_path,
        forward_horizon=args.horizon,
    )
    build_feature_panel(cfg)
    _log("=== v1 feature panel build complete ===")
    _log("You can now use RollingPanelDataset to create 240×N×F windows for modeling.")


if __name__ == "__main__":
    main()
