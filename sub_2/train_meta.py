"""
train_meta.py

FOMAMLPortfolioLearner + StablePortfolioBackbone를 이용해
메타 학습을 10 epoch 동안 수행하는 스크립트.

- train_loader: meta-train episodes
- val_loader:   meta-val episodes (간단한 평가용)
- test_loader:  아직은 사용하지 않음 (추후 실험에서 사용)

실행:
    python train_meta.py
"""

import os
import time
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim

from TaskContextEncoder import StablePortfolioBackbone as PortfolioBackbone
from TaskContextEncoder import StablePortfolioBackbone
from meta_learner import (
    FOMAMLPortfolioLearner,
    _clone_param_state,
    _load_param_state_inplace,
)


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------
def to_device_batch(batch: Dict[str, torch.Tensor], device: torch.device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------
# Evaluation (간단 버전)
# ---------------------------------------------------------------------
@torch.no_grad()
def evaluate_meta(
    meta_learner: FOMAMLPortfolioLearner,
    backbone: StablePortfolioBackbone,
    val_loader,
    device: torch.device,
    max_batches: int = 20,
) -> Dict[str, float]:
    """
    FOMAML 구조를 그대로 사용하되, gradient/optimizer step 없이
    query loss 관점에서만 평균 성능을 보는 간단 평가 함수.

    - inner-loop는 그대로 수행 (support 기반 adaptation)
    - 각 episode별 query loss를 계산해서 평균낸다.
    """
    meta_learner.eval()
    backbone.eval()

    total_q_loss = 0.0
    total_s_loss = 0.0
    total_episodes = 0

    # θ snapshot
    init_state = _clone_param_state(backbone)

    for b_idx, batch in enumerate(val_loader):
        if b_idx >= max_batches:
            break

        batch = to_device_batch(batch, device)
        support_x = batch["support_x"]
        support_y = batch["support_y"]
        query_x = batch["query_x"]
        query_y = batch["query_y"]
        asset_mask = batch["asset_mask"]

        B = support_x.shape[0]

        for i in range(B):
            # θ로 리셋
            _load_param_state_inplace(backbone, init_state)

            sx = support_x[i : i + 1]
            sy = support_y[i : i + 1]
            qx = query_x[i : i + 1]
            qy = query_y[i : i + 1]
            am = asset_mask[i : i + 1]

            # inner-loop (support 기반 적응)
            s_loss, context = meta_learner._inner_adapt(sx, sy, am)

            # query forward
            pred_query_last = backbone.forward_query(
                qx, context, return_sequence=False
            )  # (1, N)
            if not torch.isfinite(pred_query_last).all():
                pred_query_last = torch.nan_to_num(
                    pred_query_last, nan=0.0, posinf=1e6, neginf=-1e6
                )
            target_query_last = qy[:, -1, :]

            q_loss = backbone.compute_loss(
                pred_query_last,
                target_query_last,
                am,
                loss_type=meta_learner.loss_type,
            )

            total_q_loss += float(q_loss.item())
            total_s_loss += float(s_loss.item())
            total_episodes += 1

    # θ 복원
    _load_param_state_inplace(backbone, init_state)

    if total_episodes == 0:
        return {"val_query_loss": float("nan"), "val_support_loss": float("nan")}

    return {
        "val_query_loss": total_q_loss / total_episodes,
        "val_support_loss": total_s_loss / total_episodes,
    }


# ---------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------
def main():
    # -----------------------------
    # 0. 기본 설정
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 간단한 하이퍼파라미터 (필요시 config로 분리해도 됨)
    cfg = {
        "data_dir": "./processed_data",
        "k": 4,
        "tau": 0.60,
        "batch_size": 2,          # meta-batch size (episode 수)
        "min_asset_ratio": 0.60,
        "normalization_mode": "feature-wise",

        "context_dim": 32,
        "unet": {
            "base_channels": 64,
            "n_layers": 3,
            "channel_mult": None,
            "n_attention_heads": 4,
            "dropout": 0.0,       # meta-learning 안정성을 위해 0 권장
            "output_aggregation": "last",
        },
        "context_cfg": {
            "hidden_dim": 128,
            "use_returns": True,
        },

        "inner_lr": 5e-4,
        "num_inner_steps": 1,
        "soft_weight_alpha": 1.0,
        "max_inner_grad": 1.0,

        "meta_lr": 1e-4,
        "num_epochs": 10,
        "grad_clip": 1.0,
        "val_max_batches": 20,
        "ckpt_dir": "./checkpoints_meta",
    }

    os.makedirs(cfg["ckpt_dir"], exist_ok=True)

    # -----------------------------
    # 1. DataLoaders
    # -----------------------------
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=cfg["data_dir"],
        k=cfg["k"],
        tau=cfg["tau"],
        batch_size=cfg["batch_size"],
        num_workers=0,
        normalization_mode=cfg["normalization_mode"],
        ticker_subset=None,
        min_asset_ratio=cfg["min_asset_ratio"],
    )

    # 샘플 배치로 input_dim 확인
    sample_batch = next(iter(train_loader))
    B, T_s, N, F = sample_batch["support_x"].shape
    print("\n[INFO] Sample batch shapes:")
    print(f"  support_x: {sample_batch['support_x'].shape}")
    print(f"  support_y: {sample_batch['support_y'].shape}")
    print(f"  query_x:   {sample_batch['query_x'].shape}")
    print(f"  query_y:   {sample_batch['query_y'].shape}")
    print(f"  asset_mask:{sample_batch['asset_mask'].shape}")

    # -----------------------------
    # 2. Backbone & Meta-learner
    # -----------------------------
    backbone = PortfolioBackbone(
        input_dim=46,
        context_dim=64,
        unet_config=cfg["unet"],
        context_config=cfg["context_cfg"],
    ).to(device)

    n_params = sum(p.numel() for p in backbone.parameters())
    print(f"\n[INFO] StablePortfolioBackbone created (params: {n_params:,})")

    meta_learner = FOMAMLPortfolioLearner(
        backbone=backbone,
        inner_lr=1e-3,
        num_inner_steps=1,
        loss_type="mse",
        soft_weight_alpha=1.0,
        max_inner_grad=1.0,        # 필요하면 0.5 ~ 2.0 사이 튜닝
        max_meta_grad_norm=1.0,    # 0.5~5.0 정도
        skip_invalid_episodes=True # 기본값 True
    ).to(device)

    optimizer = optim.Adam(backbone.parameters(), lr=cfg["meta_lr"])

    # -----------------------------
    # 3. Training loop
    # -----------------------------
    num_epochs = cfg["num_epochs"]

    print("\n" + "=" * 80)
    print(f"Training for {num_epochs} epochs")
    print("=" * 80)

    for epoch in range(1, num_epochs + 1):
        meta_learner.train()
        backbone.train()

        epoch_meta_loss = 0.0
        epoch_support_loss = 0.0
        epoch_query_loss = 0.0
        epoch_batches = 0

        start_time = time.time()

        print(f"\n[Epoch {epoch}] --------------------------------------------")
        for batch_idx, batch in enumerate(train_loader):
            batch = to_device_batch(batch, device)

            # FOMAML meta step (optimizer step 포함)
            meta_loss, logs = meta_learner.meta_step(batch, optimizer)

            epoch_meta_loss += logs["meta_loss"]
            epoch_support_loss += logs["support_loss_mean"]
            epoch_query_loss += logs["query_loss_mean"]
            epoch_batches += 1

            # 첫 batch는 상세 로그
            if batch_idx == 0:
                print(
                    f"  [Batch {batch_idx}/{len(train_loader)}] "
                    f"meta={logs['meta_loss']:.6f}, "
                    f"support={logs['support_loss_mean']:.6f}, "
                    f"query={logs['query_loss_mean']:.6f}"
                )
                # weights도 모니터링
                print(f"    weights: {logs['weights']}")

        # Epoch 평균
        avg_meta = epoch_meta_loss / max(epoch_batches, 1)
        avg_sup = epoch_support_loss / max(epoch_batches, 1)
        avg_q = epoch_query_loss / max(epoch_batches, 1)
        elapsed = time.time() - start_time

        print(
            f"  >> Epoch {epoch} done. "
            f"meta={avg_meta:.6f}, support={avg_sup:.6f}, "
            f"query={avg_q:.6f}, time={elapsed:.1f}s"
        )

        # -----------------------------
        # 4. Validation
        # -----------------------------
        val_logs = evaluate_meta(
            meta_learner,
            backbone,
            val_loader,
            device,
            max_batches=cfg["val_max_batches"],
        )
        print(
            f"  >> Val: query_loss={val_logs['val_query_loss']:.6f}, "
            f"support_loss={val_logs['val_support_loss']:.6f}"
        )

        # -----------------------------
        # 5. Checkpoint
        # -----------------------------
        ckpt_path = os.path.join(
            cfg["ckpt_dir"],
            f"meta_epoch{epoch:03d}.pt",
        )
        torch.save(
            {
                "epoch": epoch,
                "backbone_state": backbone.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "cfg": cfg,
                "train_meta_loss": avg_meta,
                "train_support_loss": avg_sup,
                "train_query_loss": avg_q,
                "val_logs": val_logs,
            },
            ckpt_path,
        )
        print(f"  >> Checkpoint saved to: {ckpt_path}")

    print("\n" + "=" * 80)
    print("Training finished.")
    print("=" * 80)


if __name__ == "__main__":
    main()
