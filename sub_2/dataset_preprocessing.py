#!/usr/bin/env python3
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

# --------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------
# Config 로더
# --------------------------------------------------------------------
def load_data_config(path: str = "data_config.json") -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg



def load_feature_config(path: str = "config/feature_config.yaml") -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg



# --------------------------------------------------------------------
# Panel merge helpers
# --------------------------------------------------------------------
def merge_macro(panel: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """
    Merge macro panel (Date-level) into equity feature panel (Date, Ticker-level).
    Assumes:
      - panel index: MultiIndex [Date, Ticker] or similar
      - macro index: Date-like index (possibly unnamed)
    """
    logger.info("Merging macro panel into feature panel.")

    # 1) 패널 쪽: Date, Ticker 컬럼 확보
    panel_reset = panel.reset_index()
    if "Date" not in panel_reset.columns:
        raise KeyError("Feature panel must contain 'Date' after reset_index().")
    if "Ticker" not in panel_reset.columns:
        raise KeyError("Feature panel must contain 'Ticker' after reset_index().")

    # 2) 매크로 쪽: 어떤 인덱스든 상관없이 'Date' 컬럼 강제 생성
    macro_reset = macro.copy()

    # 인덱스가 DatetimeIndex / PeriodIndex인 경우: 우선 reset_index
    if isinstance(macro_reset.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        if macro_reset.index.name is None:
            macro_reset.index.name = "Date"
        macro_reset = macro_reset.reset_index()
    else:
        macro_reset = macro_reset.reset_index()

    # 그래도 'Date'가 없으면, 첫 번째 컬럼을 'Date'로 간주하고 rename
    if "Date" not in macro_reset.columns:
        first_col = macro_reset.columns[0]
        logger.warning(
            f"'Date' column not found in macro panel after reset_index(); "
            f"renaming first column '{first_col}' -> 'Date'."
        )
        macro_reset = macro_reset.rename(columns={first_col: "Date"})

    # 3) 머지
    merged = panel_reset.merge(macro_reset, on="Date", how="left")

    # 4) 다시 MultiIndex(Date, Ticker)로 복원
    merged = merged.set_index(["Date", "Ticker"]).sort_index()

    return merged



def merge_fundamentals(panel: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """
    panel: MultiIndex (Date, Ticker)
    fundamentals: MultiIndex (Date, Ticker) or columns Date,Ticker
    """
    logger.info("Merging fundamentals panel into feature panel.")
    if isinstance(fundamentals.index, pd.MultiIndex):
        fund_reset = fundamentals.reset_index()
    else:
        fund_reset = fundamentals.copy()
    panel_reset = panel.reset_index()

    merged = panel_reset.merge(
        fund_reset,
        on=["Date", "Ticker"],
        how="left",
        suffixes=("", "_fund"),
    )
    merged.sort_values(["Date", "Ticker"], inplace=True)
    merged.set_index(["Date", "Ticker"], inplace=True)
    return merged


# --------------------------------------------------------------------
# Target 생성 (예: 다음날 로그수익률, 5일 로그수익률)
# --------------------------------------------------------------------
def add_targets(panel: pd.DataFrame, close_col: str = "Close") -> pd.DataFrame:
    """
    MultiIndex (Date, Ticker) 패널에 타겟 수익률 컬럼 추가.
    - target_log_ret_1d: log(C_{t+1} / C_t)
    - target_log_ret_5d: log(C_{t+5} / C_t)
    """
    logger.info("Adding target return columns.")
    df = panel.copy()

    if close_col not in df.columns:
        raise KeyError(f"Close column '{close_col}' not found in panel columns.")

    def fwd_logret(group: pd.Series, horizon: int) -> pd.Series:
        return np.log(group.shift(-horizon) / group)

    close = df[close_col]

    df["target_log_ret_1d"] = close.groupby(level="Ticker", group_keys=False).apply(
        lambda s: fwd_logret(s, 1)
    )
    df["target_log_ret_5d"] = close.groupby(level="Ticker", group_keys=False).apply(
        lambda s: fwd_logret(s, 5)
    )

    # 타겟이 NaN인 마지막 구간은 실질적으로 학습에 쓸 수 없으므로 제거는 나중에 처리
    return df


# --------------------------------------------------------------------
# 날짜 기반 split
# --------------------------------------------------------------------
def split_by_date(
    panel: pd.DataFrame,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    test_start: str,
    test_end: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    idx_date = panel.index.get_level_values("Date")

    train_mask = (idx_date >= train_start) & (idx_date <= train_end)
    val_mask = (idx_date >= val_start) & (idx_date <= val_end)
    test_mask = (idx_date >= test_start) & (idx_date <= test_end)

    df_train = panel.loc[train_mask].copy()
    df_val = panel.loc[val_mask].copy()
    df_test = panel.loc[test_mask].copy()

    logger.info(f"Train panel shape: {df_train.shape}")
    logger.info(f"Val   panel shape: {df_val.shape}")
    logger.info(f"Test  panel shape: {df_test.shape}")

    return df_train, df_val, df_test


# --------------------------------------------------------------------
# 표준화 (train 기준 z-score)
# --------------------------------------------------------------------
def standardize_features(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, any]]:
    """
    Standardize features using train statistics (mean, std).

    1) feature_config.yaml 에서 읽어온 feature_cols 중 실제로 존재하는 컬럼만 사용.
    2) 교집합이 비면, 자동으로 numeric feature들을 감지해서 fallback.
    3) 최종적으로 사용된 feature 컬럼 리스트와 mean/std를 scaler 딕셔너리로 반환.
    """
    logger.info("Standardizing features based on train statistics.")

    # 0. 원래 설정된 feature 리스트 저장 (for logging)
    original_feature_cols = list(feature_cols)

    # 1. 실제 df_train에 존재하는 컬럼과의 교집합만 사용
    feature_cols_existing = [c for c in original_feature_cols if c in df_train.columns]

    missing = sorted(set(original_feature_cols) - set(feature_cols_existing))
    if missing:
        logger.warning(
            f"{len(missing)} feature(s) from config not found in DataFrame. "
            f"Examples: {missing[:10]}"
        )

    # 2. 교집합이 비면 numeric feature 자동 감지 fallback
    if len(feature_cols_existing) == 0:
        logger.warning(
            "No configured feature columns found in df_train. "
            "Falling back to automatic numeric feature detection."
        )

        # ID / meta 컬럼들 (표준화 제외)
        id_like_cols = {"Ticker", "ticker", "Asset", "asset"}
        # 타깃 후보 (있으면 제외)
        target_like_cols = {
            "target",
            "Target",
            "ret_1d",
            "ret_5d",
            "future_ret_1d",
            "future_ret_5d",
        }

        feature_cols_existing = [
            c
            for c in df_train.columns
            if c not in id_like_cols
            and c not in target_like_cols
            and pd.api.types.is_numeric_dtype(df_train[c])
        ]

        logger.info(
            f"Automatically detected {len(feature_cols_existing)} numeric feature columns "
            f"for standardization."
        )

    # 3. 여전히 아무 것도 없으면 명시적으로 에러
    if len(feature_cols_existing) == 0:
        raise ValueError(
            "No feature columns available for standardization. "
            "Check feature_config.yaml and DataFrame columns."
        )

    # 4. train 통계 (mean, std) 계산
    train_stats = df_train[feature_cols_existing].agg(["mean", "std"])
    mean = train_stats.loc["mean"]
    std = train_stats.loc["std"].replace(0.0, 1.0)  # 분산 0인 경우 나눗셈 방지

    scaler = {
        # Convert Series to plain dict of floats for JSON serialization downstream.
        "mean": {k: float(v) for k, v in mean.items()},
        "std": {k: float(v) for k, v in std.items()},
        "feature_cols": feature_cols_existing,
        "config_feature_cols": original_feature_cols,
    }

    def _apply_standardization(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # 없는 컬럼은 자동으로 skip 되도록 intersection
        common_cols = [c for c in feature_cols_existing if c in df.columns]
        df[common_cols] = (df[common_cols] - mean[common_cols]) / (std[common_cols] + 1e-6)
        return df

    df_train_std = _apply_standardization(df_train)
    df_val_std = _apply_standardization(df_val)
    df_test_std = _apply_standardization(df_test)

    logger.info(
        f"Standardization complete. Used {len(feature_cols_existing)} feature columns. "
        f"(Configured: {len(original_feature_cols)})"
    )

    return df_train_std, df_val_std, df_test_std, scaler



# --------------------------------------------------------------------
# NaN 처리
# --------------------------------------------------------------------
def clean_panel_for_training(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    max_feature_na_frac: float = 0.2,
) -> pd.DataFrame:
    """
    - 타겟이 NaN인 행 제거
    - feature NaN 비율이 너무 큰 행 제거 (선택)
    """
    logger.info("Cleaning panel for training (handling NaNs).")

    # 1) 타겟 NaN 제거
    before = df.shape[0]
    mask_target_valid = df[target_cols].notnull().all(axis=1)
    df = df[mask_target_valid]
    after = df.shape[0]
    logger.info(f"Dropped {before - after} rows due to NaN targets.")

    # 2) feature NaN 비율 기준 제거
    if max_feature_na_frac < 1.0:
        feat_na_frac = df[feature_cols].isna().mean(axis=1)
        mask_feat_valid = feat_na_frac <= max_feature_na_frac
        before2 = df.shape[0]
        df = df[mask_feat_valid]
        after2 = df.shape[0]
        logger.info(
            f"Dropped {before2 - after2} rows due to high feature NaN fraction "
            f"(>{max_feature_na_frac:.2f})."
        )

    # 3) 남은 NaN은 forward-fill / groupby Ticker 기반으로 처리 (보수적으로)
    df = (
        df.groupby(level="Ticker", group_keys=False)
        .apply(lambda g: g.sort_index(level="Date").ffill())
        .sort_index()
    )

    return df


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main():
    # --------------------------------------------------------------
    # 1. Config 로드
    # --------------------------------------------------------------
    data_cfg = load_data_config("config/data_config.json")
    feat_cfg = load_feature_config("config/feature_config.yaml")

    data_paths = data_cfg.get("data", {})
    date_cfg = data_cfg.get("dates", {})
    sampling_cfg = data_cfg.get("sampling", {})

    feature_panel_path = data_paths.get("feature_panel_path", "data/feature_panel.parquet")
    macro_panel_path = data_paths.get("macro_panel_path", "data/macro_panel.parquet")
    fundamentals_path = data_paths.get("fundamentals_path", "data/fundamentals_panel.parquet")

    output_dir = Path(data_paths.get("output_dir", "data/processed"))
    output_dir.mkdir(parents=True, exist_ok=True)

    train_start = date_cfg.get("train_start", "2005-01-01")
    train_end = date_cfg.get("train_end", "2015-12-31")
    val_start = date_cfg.get("val_start", "2016-01-01")
    val_end = date_cfg.get("val_end", "2019-12-31")
    test_start = date_cfg.get("test_start", "2020-01-01")
    test_end = date_cfg.get("test_end", "2025-12-31")

    # feature_config에서 feature 목록/target 목록 읽기
    features_cfg = feat_cfg.get("features", {})
    price_features = features_cfg.get("price", [])
    technical_features = features_cfg.get("technical", [])
    macro_features = features_cfg.get("macro", [])
    fundamental_features = features_cfg.get("fundamental", [])

    targets_cfg = feat_cfg.get("targets", ["target_log_ret_1d", "target_log_ret_5d"])

    # --------------------------------------------------------------
    # 2. 패널 로드
    # --------------------------------------------------------------
    logger.info(f"Loading feature panel from {feature_panel_path}")
    panel = pd.read_parquet(feature_panel_path)

    if not isinstance(panel.index, pd.MultiIndex):
        # 안전장치: 혹시 index가 Date 하나만이면 Ticker 컬럼 붙어있을 수 있음
        if {"Date", "Ticker"}.issubset(panel.columns):
            panel.set_index(["Date", "Ticker"], inplace=True)
        else:
            raise ValueError("Feature panel must have MultiIndex (Date, Ticker).")

    panel.index.set_names(["Date", "Ticker"], inplace=True)
    panel.sort_index(inplace=True)

    logger.info(f"Feature panel shape: {panel.shape}")

    logger.info(f"Loading macro panel from {macro_panel_path}")
    macro = pd.read_parquet(macro_panel_path)
    if not isinstance(macro.index, pd.DatetimeIndex):
        macro.index = pd.to_datetime(macro.index)
    macro.sort_index(inplace=True)

    # --------------------------------------------------------------
    # 3. Macro merge
    # --------------------------------------------------------------
    panel = merge_macro(panel, macro)

    # --------------------------------------------------------------
    # 4. Fundamentals (optional)
    # --------------------------------------------------------------
    if os.path.exists(fundamentals_path):
        logger.info(f"Loading fundamentals panel from {fundamentals_path}")
        fundamentals = pd.read_parquet(fundamentals_path)
        if not isinstance(fundamentals.index, pd.MultiIndex):
            if {"Date", "Ticker"}.issubset(fundamentals.columns):
                fundamentals.set_index(["Date", "Ticker"], inplace=True)
            else:
                raise ValueError("Fundamentals panel must have (Date, Ticker) information.")
        fundamentals.index.set_names(["Date", "Ticker"], inplace=True)
        fundamentals.sort_index(inplace=True)

        panel = merge_fundamentals(panel, fundamentals)
    else:
        logger.warning(f"Fundamentals file not found at {fundamentals_path}; skipping merge.")

    # --------------------------------------------------------------
    # 5. 타겟 수익률 생성
    # --------------------------------------------------------------
    panel = add_targets(panel, close_col="Close")

    # --------------------------------------------------------------
    # 6. 날짜 기반 split
    # --------------------------------------------------------------
    df_train, df_val, df_test = split_by_date(
        panel,
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
        test_end=test_end,
    )

    # --------------------------------------------------------------
    # 7. feature / target 분리 + NaN 처리
    # --------------------------------------------------------------
    # feature 후보: config에 정의된 price + technical + macro + fundamental
    all_feature_candidates = (
        list(price_features)
        + list(technical_features)
        + list(macro_features)
        + list(fundamental_features)
    )

    # 실제 panel에 존재하는 컬럼만 사용
    feature_cols = [c for c in all_feature_candidates if c in panel.columns]
    missing_feats = [c for c in all_feature_candidates if c not in panel.columns]
    if missing_feats:
        logger.warning(f"The following configured features are missing in panel and will be ignored: {missing_feats}")

    target_cols = [c for c in targets_cfg if c in panel.columns]
    missing_targets = [c for c in targets_cfg if c not in panel.columns]
    if missing_targets:
        logger.warning(f"The following configured targets are missing in panel and will be ignored: {missing_targets}")
    if not target_cols:
        raise ValueError("No valid target columns found in panel; check feature_config.yaml targets.")

    df_train = clean_panel_for_training(df_train, feature_cols, target_cols)
    df_val = clean_panel_for_training(df_val, feature_cols, target_cols)
    df_test = clean_panel_for_training(df_test, feature_cols, target_cols)

    # --------------------------------------------------------------
    # 8. 표준화 (train 기준)
    # --------------------------------------------------------------
    df_train_std, df_val_std, df_test_std, scaler = standardize_features(
        df_train, df_val, df_test, feature_cols
    )

    # --------------------------------------------------------------
    # 9. 저장
    # --------------------------------------------------------------
    train_path = output_dir / "panel_train.parquet"
    val_path = output_dir / "panel_val.parquet"
    test_path = output_dir / "panel_test.parquet"
    full_path = output_dir / "panel_full.parquet"
    scaler_path = output_dir / "feature_scaler.json"

    # full panel도 표준화 버전으로 재구성
    # (필요하면 나중에 panel_full에서 바로 사용)
    df_full_std = pd.concat(
        [df_train_std, df_val_std, df_test_std],
        axis=0,
    ).sort_index()

    logger.info(f"Saving standardized train panel to {train_path}")
    df_train_std.to_parquet(train_path)

    logger.info(f"Saving standardized val panel to {val_path}")
    df_val_std.to_parquet(val_path)

    logger.info(f"Saving standardized test panel to {test_path}")
    df_test_std.to_parquet(test_path)

    logger.info(f"Saving standardized full panel to {full_path}")
    df_full_std.to_parquet(full_path)

    logger.info(f"Saving feature scaler to {scaler_path}")
    with open(scaler_path, "w") as f:
        json.dump(scaler, f, indent=2)

    logger.info("dataset_preprocessing.py finished successfully.")


if __name__ == "__main__":
    main()
