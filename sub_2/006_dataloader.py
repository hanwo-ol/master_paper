# 006_dataloader.py
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
import warnings


class SP500MetaLearningDataset(Dataset):
    """
    S&P 500 Meta-Learning Dataset

    ⚠️ Normalization 사용 시 주의사항
    - Train split에서만 정규화 통계를 계산합니다
    - Val/Test split은 반드시 load_normalization_stats()로
      train 통계를 주입해야 합니다
    - configs/*.yaml + create_dataloaders_from_config() 조합을 쓰면 자동 처리됩니다

    Episode 구조:
    - Support: (n_support_days, n_assets, n_features)
    - Query:   (n_query_days, n_assets, n_features)
    - Support labels: (n_support_days, n_assets)
    - Query labels:   (n_query_days, n_assets)

    Asset Universe:
    - 모든 episode는 동일한 asset universe를 유지합니다
    - Support와 Query는 완전히 동일한 ticker 집합을 가집니다
    """

    def __init__(
        self,
        panel_path: str,
        episodes_path: str,
        split: str = "train",
        feature_cols: Optional[List[str]] = None,
        normalize: bool = True,
        normalization_mode: str = "feature-wise",
        ticker_subset: Optional[List[str]] = None,
        return_regime_tensor: bool = True,
        min_asset_ratio: float = 0.6,  # 최소 자산 비율 (universe 대비)
    ):
        super().__init__()

        self.panel_path = panel_path
        self.episodes_path = episodes_path
        self.split = split
        self.normalize = normalize
        self.normalization_mode = normalization_mode
        self.ticker_subset = ticker_subset
        self.return_regime_tensor = return_regime_tensor
        self.min_asset_ratio = min_asset_ratio

        # 데이터 로드
        print(f"Loading dataset for split: {split}")
        self._load_panel_data()
        self._load_episodes()
        self._setup_features(feature_cols)

        if self.normalize:
            self._compute_normalization_stats()

        print(f"✓ Dataset ready: {len(self)} episodes")
        print(f"  Features: {len(self.feature_cols)}")
        print(f"  Assets:   {len(self.asset_universe)}")

    def _load_panel_data(self):
        """Panel 데이터 로드"""
        print(f"  Loading panel: {self.panel_path}")
        self.panel_data = pd.read_csv(self.panel_path, index_col=0, parse_dates=True)

        # Ticker subset 필터링
        if self.ticker_subset is not None:
            print(f"  Filtering to {len(self.ticker_subset)} tickers")
            self.panel_data = self.panel_data[
                self.panel_data["ticker"].isin(self.ticker_subset)
            ].copy()

        # Asset universe 고정 (알파벳 순서)
        self.asset_universe = sorted(self.panel_data["ticker"].unique())

        # Ticker → index 매핑
        self.ticker_to_idx = {
            ticker: idx for idx, ticker in enumerate(self.asset_universe)
        }

        print(f"  Panel shape: {self.panel_data.shape}")
        print(f"  Asset universe: {len(self.asset_universe)} tickers")

    def _load_episodes(self):
        """Episodes 로드 및 검증"""
        print(f"  Loading episodes: {self.episodes_path}")
        with open(self.episodes_path, "r") as f:
            all_episodes = json.load(f)

        # Split 필터링
        raw_episodes = [ep for ep in all_episodes if ep["split"] == self.split]
        print(f"  Raw episodes for {self.split}: {len(raw_episodes)}")

        # Episode 검증 및 필터링
        valid_episodes = []
        skipped_reasons = {"min_assets": 0, "ticker_mismatch": 0, "other": 0}

        for ep in raw_episodes:
            valid, reason = self._validate_episode(ep)
            if valid:
                valid_episodes.append(ep)
            else:
                skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1

        self.episodes = valid_episodes
        print(f"  Valid episodes: {len(self.episodes)}")

        if sum(skipped_reasons.values()) > 0:
            print("  Skipped:")
            for reason, count in skipped_reasons.items():
                if count > 0:
                    print(f"    - {reason}: {count}")

        # Regime 분포
        regime_counts = {}
        for ep in self.episodes:
            regime = ep["support_regime"]
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        print("  Regime distribution:")
        for regime, count in sorted(regime_counts.items()):
            pct = count / len(self.episodes) * 100
            print(f"    Regime {regime}: {count} ({pct:.1f}%)")

    def _validate_episode(self, episode: Dict) -> Tuple[bool, str]:
        """
        Episode 검증

        Returns:
            (is_valid, skip_reason)
        """
        support_row_ids = episode["support_row_ids"]
        query_row_ids = episode["query_row_ids"]

        support_data = self.panel_data[
            self.panel_data["row_id"].isin(support_row_ids)
        ]
        query_data = self.panel_data[self.panel_data["row_id"].isin(query_row_ids)]

        support_tickers = set(support_data["ticker"].unique())
        query_tickers = set(query_data["ticker"].unique())

        # 최소 자산 수 확인
        min_assets = max(10, int(len(self.asset_universe) * self.min_asset_ratio))

        if len(support_tickers) < min_assets:
            return False, "min_assets"

        if len(query_tickers) < min_assets:
            return False, "min_assets"

        # Support와 Query가 완전히 동일한 asset universe를 가지는지 확인
        if support_tickers != query_tickers:
            return False, "ticker_mismatch"

        return True, "valid"

    def _setup_features(self, feature_cols: Optional[List[str]]):
        """Feature 컬럼 설정"""
        if feature_cols is not None:
            self.feature_cols = feature_cols
        else:
            # 자동 감지
            exclude = {"ticker", "returns", "row_id", "regime"}
            self.feature_cols = [
                col for col in self.panel_data.columns if col not in exclude
            ]

        print(f"  Features ({len(self.feature_cols)}):")

        # Feature 카테고리 분류
        self.feature_groups = {
            "price": [
                c
                for c in self.feature_cols
                if c in ["open", "high", "low", "close", "volume"]
            ],
            "technical": [
                c
                for c in self.feature_cols
                if c
                in [
                    "rsi",
                    "ma_20",
                    "ma_50",
                    "ma_200",
                    "macd",
                    "macd_signal",
                    "bollinger_upper",
                    "bollinger_lower",
                    "atr",
                    "stoch_k",
                    "stoch_d",
                    "price_roc",
                    "realized_vol_20",
                    "realized_vol_60",
                    "volume_roc",
                    "mfi",
                    "obv",
                    "williams_r",
                ]
            ],
            "fundamental": [
                c
                for c in self.feature_cols
                if any(x in c for x in ["pe_", "pb_", "roe", "roa", "debt", "dividend", "payout", "cashflow"])
            ],
            "macro": [
                c
                for c in self.feature_cols
                if c
                in [
                    "vix",
                    "treasury_10y",
                    "treasury_2y",
                    "yield_spread",
                    "usd_index",
                    "credit_spread",
                    "cpi_yoy",
                ]
            ],
            "missing": [c for c in self.feature_cols if c.endswith("_missing")],
        }

        for group, cols in self.feature_groups.items():
            if len(cols) > 0:
                print(f"    - {group.capitalize()}: {len(cols)}")

    def _compute_normalization_stats(self):
        """정규화 통계 계산 (Train만) + 의미 없는 feature 명시적 처리"""
        if self.split != 'train':
            # Val/Test는 여기서 통계 계산 안 함
            self.feature_mean = None
            self.feature_std = None
            self.per_asset_stats = None
            print(f"  ⚠ Normalization stats not computed for split={self.split}")
            print(f"     Use load_normalization_stats() to inject train stats")
            return

        if self.normalization_mode == 'feature-wise':
            # ---------- Feature-wise 정규화 ----------
            train_values = self.panel_data[self.feature_cols].values.astype(np.float32)
            n_features = len(self.feature_cols)

            means = np.zeros(n_features, dtype=np.float32)
            stds = np.ones(n_features, dtype=np.float32)

            degenerate_features = []  # (name, reason) 리스트

            for j, col_name in enumerate(self.feature_cols):
                col = train_values[:, j]
                # 유효한 값 (NaN/inf 제외)
                valid_mask = np.isfinite(col)
                n_valid = valid_mask.sum()

                if n_valid == 0:
                    # 1) 전 구간에서 관측값이 한 개도 없는 feature
                    degenerate_features.append((col_name, "no_valid_observations"))
                    means[j] = 0.0
                    stds[j] = 1.0
                    continue

                col_valid = col[valid_mask]
                m = float(col_valid.mean())
                s = float(col_valid.std())

                if s < 1e-6:
                    # 2) 분산이 0에 가까운 feature (상수형)
                    degenerate_features.append((col_name, "zero_or_near_zero_variance"))
                    means[j] = 0.0
                    stds[j] = 1.0
                else:
                    means[j] = m
                    stds[j] = s

            self.feature_mean = torch.tensor(means, dtype=torch.float32)
            self.feature_std = torch.tensor(stds, dtype=torch.float32)

            print(f"  Normalization: feature-wise (computed from train)")

            if len(degenerate_features) > 0:
                print("  [Normalization] Degenerate features detected → set to mean=0, std=1")
                for name, reason in degenerate_features:
                    print(f"    - {name}: {reason}")

        elif self.normalization_mode == 'per-asset':
            # ---------- 자산별 정규화 ----------
            stats = []
            degenerate_pairs = []  # (ticker, feature, reason)

            for ticker in self.asset_universe:
                ticker_data = self.panel_data[
                    self.panel_data['ticker'] == ticker
                ][self.feature_cols].values.astype(np.float32)

                n_features = len(self.feature_cols)
                means = np.zeros(n_features, dtype=np.float32)
                stds = np.ones(n_features, dtype=np.float32)

                for j, col_name in enumerate(self.feature_cols):
                    col = ticker_data[:, j]
                    valid_mask = np.isfinite(col)
                    n_valid = valid_mask.sum()

                    if n_valid == 0:
                        degenerate_pairs.append((ticker, col_name, "no_valid_observations"))
                        means[j] = 0.0
                        stds[j] = 1.0
                        continue

                    col_valid = col[valid_mask]
                    m = float(col_valid.mean())
                    s = float(col_valid.std())

                    if s < 1e-6:
                        degenerate_pairs.append((ticker, col_name, "zero_or_near_zero_variance"))
                        means[j] = 0.0
                        stds[j] = 1.0
                    else:
                        means[j] = m
                        stds[j] = s

                stats.append({'mean': means, 'std': stds})

            self.per_asset_stats = stats
            print(f"  Normalization: per-asset (computed from train)")

            if len(degenerate_pairs) > 0:
                # 너무 많이 찍히면 로그가 지저분해질 수 있어서 개수만 요약
                print("  [Normalization] Degenerate (ticker, feature) pairs detected")
                print(f"    Total: {len(degenerate_pairs)} pairs")
                # 필요하면 일부 샘플도 찍어줄 수 있음
                sample_n = min(10, len(degenerate_pairs))
                for ticker, feat, reason in degenerate_pairs[:sample_n]:
                    print(f"    - {ticker}, {feat}: {reason}")
                if len(degenerate_pairs) > sample_n:
                    print(f"    ... (+{len(degenerate_pairs) - sample_n} more)")

        else:
            # cross-sectional의 경우 runtime에서 처리
            print(f"  Normalization: cross-sectional (runtime normalization)")


    def load_normalization_stats(self, stats: Dict):
        """외부 정규화 통계 로드 (Val/Test용)"""
        if self.normalization_mode == "feature-wise":
            self.feature_mean = stats["mean"]
            self.feature_std = stats["std"]
        elif self.normalization_mode == "per-asset":
            self.per_asset_stats = stats["per_asset_stats"]
        print(f"  ✓ Normalization stats loaded ({self.normalization_mode})")

    def get_normalization_stats(self) -> Dict:
        """정규화 통계 반환 (Train만 호출 가능)"""
        if self.split != "train":
            raise RuntimeError(f"Cannot get normalization stats from {self.split} split")

        if self.normalization_mode == "feature-wise":
            if self.feature_mean is None or self.feature_std is None:
                raise RuntimeError("Normalization stats not computed")
            return {"mean": self.feature_mean, "std": self.feature_std}
        elif self.normalization_mode == "per-asset":
            if self.per_asset_stats is None:
                raise RuntimeError("Per-asset stats not computed")
            return {"per_asset_stats": self.per_asset_stats}
        return {}

    def _reshape_to_episode_format(
        self, data: pd.DataFrame, expected_days: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Panel data를 (Days, Assets, Features) 형태로 reshape

        Args:
            data: Panel DataFrame (sorted by date)
            expected_days: 예상 episode 길이 (검증용)

        Returns:
            x: (actual_days, n_assets, n_features)
            y: (actual_days, n_assets)
        """
        # 날짜별로 그룹화
        dates = sorted(data.index.unique())
        actual_days = len(dates)

        # 예상 길이와 실제 길이 검증
        if actual_days != expected_days:
            warnings.warn(
                f"Expected {expected_days} days, but got {actual_days} days. "
                f"This may happen due to missing trading days."
            )

        # 초기화
        n_assets = len(self.asset_universe)
        n_features = len(self.feature_cols)

        x = np.full((actual_days, n_assets, n_features), np.nan, dtype=np.float32)
        y = np.full((actual_days, n_assets), np.nan, dtype=np.float32)

        # 날짜별로 데이터 채우기
        for day_idx, date in enumerate(dates):
            day_data = data[data.index == date]

            for _, row in day_data.iterrows():
                ticker = row["ticker"]
                if ticker in self.ticker_to_idx:
                    asset_idx = self.ticker_to_idx[ticker]

                    # Features
                    x[day_idx, asset_idx, :] = row[self.feature_cols].values

                    # Returns
                    y[day_idx, asset_idx] = row["returns"]

        # Tensor 변환
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return x, y

    def _normalize_episode(self, x: torch.Tensor) -> torch.Tensor:
        """Episode 정규화"""
        if not self.normalize:
            return x

        if self.normalization_mode == "feature-wise":
            if self.feature_mean is None or self.feature_std is None:
                raise RuntimeError("Normalization stats not loaded")

            # (days, assets, features) → feature별 정규화
            x = (x - self.feature_mean.view(1, 1, -1)) / self.feature_std.view(
                1, 1, -1
            )

        elif self.normalization_mode == "per-asset":
            if self.per_asset_stats is None:
                raise RuntimeError("Per-asset stats not loaded")

            # 자산별 정규화
            for asset_idx in range(len(self.asset_universe)):
                stats = self.per_asset_stats[asset_idx]
                mean = torch.tensor(stats["mean"], dtype=torch.float32)
                std = torch.tensor(stats["std"], dtype=torch.float32)

                x[:, asset_idx, :] = (x[:, asset_idx, :] - mean) / std

        elif self.normalization_mode == "cross-sectional":
            # Cross-sectional: 각 날짜, 각 feature에 대해 자산 간 정규화
            for day_idx in range(x.shape[0]):
                for feat_idx in range(x.shape[2]):
                    values = x[day_idx, :, feat_idx]
                    valid_mask = ~torch.isnan(values)

                    if valid_mask.sum() > 1:
                        mean = values[valid_mask].mean()
                        std = values[valid_mask].std()
                        if std > 1e-6:
                            x[day_idx, :, feat_idx] = (values - mean) / std

        return x

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        단일 episode 반환

        Returns:
            {
                'support_x': (n_support_days, n_assets, n_features),
                'support_y': (n_support_days, n_assets),
                'query_x': (n_query_days, n_assets, n_features),
                'query_y': (n_query_days, n_assets),
                'support_regime': (1,) or int,
                'query_regime': (1,) or int,
                'episode_id': int,
                'n_assets': int,
                'asset_mask': (n_assets,) bool tensor
            }
        """
        episode = self.episodes[idx]

        # Support set
        support_row_ids = episode["support_row_ids"]
        support_data = self.panel_data[
            self.panel_data["row_id"].isin(support_row_ids)
        ].sort_index()

        support_x, support_y = self._reshape_to_episode_format(
            support_data, expected_days=episode["n_support_days"]
        )

        # Query set
        query_row_ids = episode["query_row_ids"]
        query_data = self.panel_data[
            self.panel_data["row_id"].isin(query_row_ids)
        ].sort_index()

        query_x, query_y = self._reshape_to_episode_format(
            query_data, expected_days=episode["n_query_days"]
        )

        # ✅ Asset mask 계산 (NaN 처리 전에)
        support_present = ~torch.isnan(support_y).all(dim=0)  # (n_assets,)
        query_present = ~torch.isnan(query_y).all(dim=0)      # (n_assets,)
        asset_mask = (support_present & query_present)

        # 정규화
        support_x = self._normalize_episode(support_x)
        query_x = self._normalize_episode(query_x)

        # NaN → 0 (mask 계산 후)
        support_x = torch.nan_to_num(support_x, nan=0.0)
        query_x = torch.nan_to_num(query_x, nan=0.0)
        support_y = torch.nan_to_num(support_y, nan=0.0)
        query_y = torch.nan_to_num(query_y, nan=0.0)

        # Regime tensor
        if self.return_regime_tensor:
            support_regime = torch.tensor(
                [episode["support_regime"]], dtype=torch.long
            )
            query_regime = torch.tensor([episode["query_regime"]], dtype=torch.long)
        else:
            support_regime = episode["support_regime"]
            query_regime = episode["query_regime"]

        return {
            "support_x": support_x,
            "support_y": support_y,
            "query_x": query_x,
            "query_y": query_y,
            "support_regime": support_regime,
            "query_regime": query_regime,
            "episode_id": episode["episode_id"],
            "n_assets": len(self.asset_universe),
            "asset_mask": asset_mask,
            "support_purity": episode["support_purity"],
            "query_purity": episode["query_purity"],
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Batch collation (batch_size > 1 지원)

    Episode 길이가 다를 수 있으므로 padding 처리
    """
    if len(batch) == 1:
        # Batch size = 1: 차원만 추가
        item = batch[0]
        return {
            "support_x": item["support_x"].unsqueeze(0),
            "support_y": item["support_y"].unsqueeze(0),
            "query_x": item["query_x"].unsqueeze(0),
            "query_y": item["query_y"].unsqueeze(0),
            "support_regime": item["support_regime"].unsqueeze(0)
            if isinstance(item["support_regime"], torch.Tensor)
            else torch.tensor([[item["support_regime"]]], dtype=torch.long),
            "query_regime": item["query_regime"].unsqueeze(0)
            if isinstance(item["query_regime"], torch.Tensor)
            else torch.tensor([[item["query_regime"]]], dtype=torch.long),
            "asset_mask": item["asset_mask"].unsqueeze(0),
            "n_assets": item["n_assets"],
            "episode_id": [item["episode_id"]],
        }

    # Batch size > 1: padding 필요
    max_support_days = max(b["support_x"].shape[0] for b in batch)
    max_query_days = max(b["query_x"].shape[0] for b in batch)
    n_assets = batch[0]["n_assets"]
    n_features = batch[0]["support_x"].shape[2]

    batch_support_x = torch.zeros(len(batch), max_support_days, n_assets, n_features)
    batch_support_y = torch.zeros(len(batch), max_support_days, n_assets)
    batch_query_x = torch.zeros(len(batch), max_query_days, n_assets, n_features)
    batch_query_y = torch.zeros(len(batch), max_query_days, n_assets)

    batch_support_regime = torch.zeros(len(batch), 1, dtype=torch.long)
    batch_query_regime = torch.zeros(len(batch), 1, dtype=torch.long)
    batch_asset_mask = torch.zeros(len(batch), n_assets, dtype=torch.bool)
    episode_ids = []

    for i, item in enumerate(batch):
        s_days = item["support_x"].shape[0]
        q_days = item["query_x"].shape[0]

        batch_support_x[i, :s_days] = item["support_x"]
        batch_support_y[i, :s_days] = item["support_y"]
        batch_query_x[i, :q_days] = item["query_x"]
        batch_query_y[i, :q_days] = item["query_y"]

        batch_support_regime[i] = item["support_regime"]
        batch_query_regime[i] = item["query_regime"]
        batch_asset_mask[i] = item["asset_mask"]
        episode_ids.append(item["episode_id"])

    return {
        "support_x": batch_support_x,
        "support_y": batch_support_y,
        "query_x": batch_query_x,
        "query_y": batch_query_y,
        "support_regime": batch_support_regime,
        "query_regime": batch_query_regime,
        "asset_mask": batch_asset_mask,
        "n_assets": n_assets,
        "episode_id": episode_ids,
    }


def create_dataloaders(
    data_dir: str,
    k: int = 4,
    tau: float = 0.60,
    batch_size: int = 1,
    num_workers: int = 0,
    normalization_mode: str = "feature-wise",
    ticker_subset: Optional[List[str]] = None,
    min_asset_ratio: float = 0.6,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Train/Val/Test DataLoader 생성 (직접 인자 버전)

    보통은 아래의 create_dataloaders_from_config()를 쓰고,
    이 함수는 low-level 용도로 남겨두면 됨.
    """

    config_dir = os.path.join(data_dir, f"K{k}_tau{tau:.2f}")
    panel_path = os.path.join(config_dir, f"sp500_panel_K{k}.csv")
    episodes_path = os.path.join(config_dir, f"episodes_K{k}.json")

    print("=" * 80)
    print(f"Creating DataLoaders (K={k}, τ={tau:.2f})")
    print(f"Normalization: {normalization_mode}")
    print(f"Min asset ratio: {min_asset_ratio}")
    print("=" * 80)

    # Datasets
    train_dataset = SP500MetaLearningDataset(
        panel_path=panel_path,
        episodes_path=episodes_path,
        split="train",
        normalize=True,
        normalization_mode=normalization_mode,
        ticker_subset=ticker_subset,
        min_asset_ratio=min_asset_ratio,
    )

    val_dataset = SP500MetaLearningDataset(
        panel_path=panel_path,
        episodes_path=episodes_path,
        split="val",
        normalize=True,
        normalization_mode=normalization_mode,
        ticker_subset=ticker_subset,
        min_asset_ratio=min_asset_ratio,
    )

    test_dataset = SP500MetaLearningDataset(
        panel_path=panel_path,
        episodes_path=episodes_path,
        split="test",
        normalize=True,
        normalization_mode=normalization_mode,
        ticker_subset=ticker_subset,
        min_asset_ratio=min_asset_ratio,
    )

    # Normalization stats 공유 (train → val/test)
    stats = train_dataset.get_normalization_stats()
    val_dataset.load_normalization_stats(stats)
    test_dataset.load_normalization_stats(stats)

    # 항상 collate_fn 사용 (batch_size=1이어도)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print("\n✓ DataLoaders created")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val:   {len(val_loader)} batches")
    print(f"  Test:  {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


def create_dataloaders_from_config(cfg: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    YAML/JSON config 기반 DataLoader 생성 helper

    예시:
        with open("configs/experiment.yaml") as f:
            cfg = yaml.safe_load(f)

        train_loader, val_loader, test_loader = create_dataloaders_from_config(cfg)
    """
    data_cfg = cfg["data"]
    dl_cfg = cfg.get("dataloader", {})
    
    data_dir = data_cfg["data_dir"]
    k = data_cfg["regime_k"]
    tau = float(data_cfg["purity_tau"])
    min_asset_ratio = float(data_cfg.get("min_asset_ratio", 0.6))
    normalization_mode = data_cfg.get("normalization", "feature-wise")
    ticker_subset = data_cfg.get("ticker_subset", None)

    batch_size = dl_cfg.get("batch_size", 1)
    num_workers = dl_cfg.get("num_workers", 0)

    return create_dataloaders(
        data_dir=data_dir,
        k=k,
        tau=tau,
        batch_size=batch_size,
        num_workers=num_workers,
        normalization_mode=normalization_mode,
        ticker_subset=ticker_subset,
        min_asset_ratio=min_asset_ratio,
    )


# 간단 테스트용 엔트리포인트
if __name__ == "__main__":
    # 예: config 없이 직접 호출
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir="./processed_data",
        k=4,
        tau=0.60,
        batch_size=1,
    )

    print("\n샘플 확인 (batch_size=1):")
    for batch in train_loader:
        print(f"  Support X:   {batch['support_x'].shape}")
        print(f"  Support Y:   {batch['support_y'].shape}")
        print(f"  Query X:     {batch['query_x'].shape}")
        print(f"  Query Y:     {batch['query_y'].shape}")
        print(f"  Asset mask:  {batch['asset_mask'].shape} (valid={batch['asset_mask'].sum().item()})")
        print(f"  Regime:      {batch['support_regime'].item()}")
        break
