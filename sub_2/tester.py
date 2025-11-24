# 009_tester.py

import torch
from torch import nn

# 파일 이름/모듈 이름은 실제 프로젝트 구조에 맞게 수정해야 합니다.
# 예: models/008_TaskContextEncoder.py 에 있다면
# from models.TaskContextEncoder import PortfolioBackbone
from dataloader import create_dataloaders
from TaskContextEncoder import PortfolioBackbone


def main():
    # ----------------------------------------------------------------------
    # 0. 환경/설정
    # ----------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    data_dir = "./processed_data"
    k = 4
    tau = 0.60
    batch_size = 2  # 메타 배치 크기는 나중에 조정

    # ----------------------------------------------------------------------
    # 1. DataLoader 생성
    # ----------------------------------------------------------------------
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        k=k,
        tau=tau,
        batch_size=batch_size,
        normalization_mode="feature-wise",
        min_asset_ratio=0.6,
    )

    print("\n[INFO] One batch from train_loader ...")
    batch = next(iter(train_loader))

    support_x = batch["support_x"]  # (B, T_s, N, F)
    support_y = batch["support_y"]  # (B, T_s, N)
    query_x = batch["query_x"]      # (B, T_q, N, F)
    query_y = batch["query_y"]      # (B, T_q, N)
    asset_mask = batch["asset_mask"]  # (B, N)

    B, T_s, N, F = support_x.shape
    _, T_q, _, _ = query_x.shape

    print(f"  support_x: {support_x.shape}")
    print(f"  support_y: {support_y.shape}")
    print(f"  query_x:   {query_x.shape}")
    print(f"  query_y:   {query_y.shape}")
    print(f"  asset_mask: {asset_mask.shape} (valid={asset_mask.sum().item()})")

    # ----------------------------------------------------------------------
    # 2. PortfolioBackbone 생성
    # ----------------------------------------------------------------------
    context_dim = 32

    unet_config = {
        "base_channels": 64,
        "n_layers": 3,
        "channel_mult": None,         # None이면 [1,2,4]로 자동 설정
        "n_attention_heads": 4,
        "dropout": 0.1,
        "output_aggregation": "last",  # query는 마지막 시점만 사용할 것
    }

    context_config = {
        "hidden_dim": 128,
        "use_returns": True,
    }

    model = PortfolioBackbone(
        input_dim=F,
        context_dim=context_dim,
        unet_config=unet_config,
        context_config=context_config,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[INFO] PortfolioBackbone created (params: {n_params:,})")

    # ----------------------------------------------------------------------
    # 3. Forward: support + query
    # ----------------------------------------------------------------------
    print("[DEBUG] support_x finite:", torch.isfinite(support_x).all())
    print("[DEBUG] support_y finite:", torch.isfinite(support_y).all())
    print("[DEBUG] query_x finite:", torch.isfinite(query_x).all())
    print("[DEBUG] query_y finite:", torch.isfinite(query_y).all())


    support_x = support_x.to(device)
    support_y = support_y.to(device)
    query_x = query_x.to(device)
    query_y = query_y.to(device)
    asset_mask = asset_mask.to(device)

    model.eval()
    with torch.no_grad():
        # 3-1) Support forward (full sequence)
        print("\n[STEP] Forward on support set")
        pred_support_seq, context = model.forward_support(
            support_x=support_x,
            support_y=support_y,
            asset_mask=asset_mask,
            return_sequence=True,    # (B, T_s, N)
        )
        
        print(f"  pred_support_seq: {pred_support_seq.shape}")
        print(f"  context:          {context.shape}")
        print(f"  context stats: mean={context.mean().item():.4f}, std={context.std().item():.4f}")

        # 3-2) Support loss
        print("\n[STEP] Compute support loss (MSE)")
        loss_support = model.compute_loss(
            pred=pred_support_seq,
            target=support_y,
            asset_mask=asset_mask,
            loss_type="mse",
        )
        print(f"  support loss: {loss_support.item():.6f}")

        # 3-3) Query forward (마지막 timestep만)
        print("\n[STEP] Forward on query set (using same context)")
        pred_query_last = model.forward_query(
            query_x=query_x,
            context=context,
            return_sequence=False,   # (B, N)
        )
        print(f"  pred_query_last: {pred_query_last.shape}  (expected: ({B}, {N}))")

        # 3-4) Query loss (마지막 시점)
        print("\n[STEP] Compute query loss (MSE, last timestep)")
        target_query_last = query_y[:, -1, :]  # (B, N)
        loss_query = model.compute_loss(
            pred=pred_query_last,
            target=target_query_last,
            asset_mask=asset_mask,
            loss_type="mse",
        )
        print(f"  query loss: {loss_query.item():.6f}")

    print("\n[RESULT] 006_dataloader ↔ PortfolioBackbone 호환성 테스트 완료")


if __name__ == "__main__":
    main()
