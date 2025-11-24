import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings


class FiLM2d(nn.Module):
    """FiLM (Feature-wise Linear Modulation) layer for 2D feature maps.

    Applies affine transformation conditioned on regime embedding:
        out = gamma * h + beta

    Args:
        embed_dim: Dimension of regime embedding (D).
        num_features: Number of channels in feature map (C).
        hidden_mult: Multiplier for hidden layer size.
        gamma_clip: Clamp range for gamma in [-gamma_clip, gamma_clip].
        beta_clip: Clamp range for beta in [-beta_clip, beta_clip].

    Input:
        h: (B, C, N, T) - Feature map
        regime_embed: (B, D) - Regime embedding

    Output:
        (B, C, N, T) - Modulated feature map
    """

    def __init__(
        self,
        embed_dim: int,
        num_features: int,
        hidden_mult: int = 2,
        gamma_clip: float = 5.0,
        beta_clip: float = 5.0,
    ):
        super().__init__()
        hidden_dim = num_features * hidden_mult
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * num_features),
        )
        self.gamma_clip = gamma_clip
        self.beta_clip = beta_clip

    def forward(self, h: torch.Tensor, regime_embed: torch.Tensor) -> torch.Tensor:
        B, C, N, T = h.shape
        gamma_beta = self.mlp(regime_embed)  # (B, 2C)
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # Each (B, C)

        # 안정화: gamma/beta를 적당한 범위로 클램프
        gamma = torch.clamp(gamma, -self.gamma_clip, self.gamma_clip)
        beta = torch.clamp(beta, -self.beta_clip, self.beta_clip)

        gamma = gamma.view(B, C, 1, 1)
        beta = beta.view(B, C, 1, 1)
        return gamma * h + beta


def _make_conv_block(in_ch: int, out_ch: int, dropout: float = 0.0) -> nn.Sequential:
    """Create a convolutional block with two conv layers.

    Structure: Conv2d -> GroupNorm -> GELU -> Conv2d -> GroupNorm -> GELU -> [Dropout]

    Spatial dimensions: (N, T) where we only pool/upsample along T.

    Args:
        in_ch: Input channels
        out_ch: Output channels
        dropout: Dropout probability (0 = no dropout)

    Returns:
        Sequential module containing the conv block
    """
    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.GroupNorm(num_groups=8, num_channels=out_ch),
        nn.GELU(),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.GroupNorm(num_groups=8, num_channels=out_ch),
        nn.GELU(),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class TimeSeriesUNet(nn.Module):
    """HRP-aware time-asset U-Net with bottleneck attention.

    This architecture implements the portfolio optimization network from the thesis:
      - Input: X ∈ R^{B×T×N×F} (batch, time, assets, features)
      - Convolutions operate over (assets, time) so HRP ordering matters
      - Encoder: conv blocks + max-pool along time dimension
      - Bottleneck: multi-head self-attention over time
      - Decoder: transposed conv upsampling + skip concatenation
      - Regime conditioning: FiLM on all encoder/decoder blocks

    Hyperparameter Guidance (from paper experiments):
        - n_layers=3 balances depth and training stability
        - base_channels=64 sufficient for N=200 assets
        - n_attention_heads=4 recommended for bottleneck
        - dropout>0.1 may hurt performance in this setting

    Args:
        input_dim: Number of per-asset features F (typically 40-60)
        base_channels: Number of channels in first conv stage (32-128 recommended)
        n_layers: Depth of U-Net (2-4 optimal, >=2 required)
        channel_mult: Channel multipliers per stage; if None, uses [2^i for i in range(n_layers)]
        regime_embed_dim: Dimensionality of regime embedding (16-64 typical)
        n_attention_heads: Number of heads in bottleneck self-attention ({2,4,8})
        dropout: Dropout rate inside conv blocks (0.0-0.2)
        output_aggregation: How to aggregate time dimension ('last', 'mean', 'attention')
        use_positional_encoding: Whether to add positional encoding on time axis
        max_time_len: Maximum time length for positional encoding buffer
        max_assets: Maximum number of assets (for attention pooling initialization)
    """

    def __init__(
        self,
        input_dim: int = 50,
        base_channels: int = 64,
        n_layers: int = 3,
        channel_mult: list[int] | None = None,
        regime_embed_dim: int = 32,
        n_attention_heads: int = 4,
        dropout: float = 0.0,
        output_aggregation: str = 'last',
        use_positional_encoding: bool = False,
        max_time_len: int = 512,
        max_assets: int = 200,
    ) -> None:
        super().__init__()

        # Validation
        assert n_layers >= 2, "UNet depth n_layers should be >= 2"
        assert output_aggregation in ['last', 'mean', 'attention'], \
            f"output_aggregation must be 'last', 'mean', or 'attention', got {output_aggregation}"

        if channel_mult is None:
            channel_mult = [2**i for i in range(n_layers)]
        assert len(channel_mult) == n_layers, \
            f"channel_mult length ({len(channel_mult)}) must equal n_layers ({n_layers})"

        # Store hyperparameters
        self.input_dim = input_dim
        self.base_channels = base_channels
        self.n_layers = n_layers
        self.channel_mult = channel_mult
        self.output_aggregation = output_aggregation
        self.use_positional_encoding = use_positional_encoding

        # Channels per stage
        channels = [base_channels * m for m in channel_mult]
        self.channels = channels

        # === Input projection ===
        # Transform (B, T, N, F) -> (B, F, N, T) -> (B, C0, N, T)
        self.input_proj = nn.Conv2d(input_dim, channels[0], kernel_size=1)

        # === Regime embedding projection ===
        self.regime_proj = nn.Linear(regime_embed_dim, regime_embed_dim)

        # === Optional positional encoding ===
        if self.use_positional_encoding:
            self.pos_encoding = nn.Parameter(
                torch.randn(1, 1, 1, max_time_len) * 0.02
            )

        # === Optional attention pooling for output ===
        if self.output_aggregation == 'attention':
            embed_dim = max_assets
            # Find valid number of heads (must divide embed_dim)
            num_heads = 4 if embed_dim % 4 == 0 else 2 if embed_dim % 2 == 0 else 1
            self.attn_pool = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                batch_first=True,
            )
        else:
            self.attn_pool = None

        # === Encoder blocks + FiLM ===
        self.encoders = nn.ModuleList()
        self.film_enc = nn.ModuleList()

        in_ch = channels[0]
        for c in channels:
            self.encoders.append(_make_conv_block(in_ch, c, dropout=dropout))
            # FiLM2d에는 clamp 파라미터가 기본값으로 들어있음
            self.film_enc.append(FiLM2d(regime_embed_dim, c))
            in_ch = c

        # === Bottleneck self-attention ===
        self.bottleneck_attn = nn.MultiheadAttention(
            embed_dim=channels[-1],
            num_heads=n_attention_heads,
            batch_first=True,
        )
        # Gating parameter for attention residual (sigmoid(0) = 0.5 initially)
        self._lambda_gate = nn.Parameter(torch.tensor(0.0))

        # === Decoder blocks + FiLM ===
        self.decoders = nn.ModuleList()
        self.film_dec = nn.ModuleList()

        # n_layers encoder stages but only n_layers-1 upsampling stages
        for i in reversed(range(n_layers - 1)):
            in_ch = channels[i + 1]
            out_ch = channels[i]

            # Transposed conv upsamples only along time (stride (1,2))
            up = nn.ConvTranspose2d(
                in_ch,
                out_ch,
                kernel_size=(1, 2),
                stride=(1, 2),
            )
            # Conv block after concatenating skip features (2*out_ch -> out_ch)
            conv = _make_conv_block(out_ch * 2, out_ch, dropout=dropout)

            self.decoders.append(nn.ModuleDict({"up": up, "conv": conv}))
            self.film_dec.append(FiLM2d(regime_embed_dim, out_ch))

        # === Final output head ===
        self.out_head = nn.Conv2d(channels[0], 1, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        regime_embed: torch.Tensor,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features (B, T, N, F)
            regime_embed: Regime embedding (B, D_regime)
            return_sequence: If True, returns full time sequence (B, T_out, N)
                           If False, returns aggregated output (B, N)

        Returns:
            Predicted returns (B, N) or (B, T_out, N) depending on return_sequence
        """
        B, T, N, F_in = x.shape
        assert F_in == self.input_dim, \
            f"Expected input_dim={self.input_dim}, got {F_in}"

        # === Input transformation ===
        # (B, T, N, F) -> (B, F, N, T)
        x = x.permute(0, 3, 2, 1)
        h = self.input_proj(x)  # (B, C0, N, T)

        # Project regime embedding
        regime_embed = self.regime_proj(regime_embed)  # (B, D)

        # Optional positional encoding on time axis
        if hasattr(self, "pos_encoding") and self.use_positional_encoding:
            pe = self.pos_encoding[..., : h.shape[-1]]
            h = h + pe

        # === Encoder ===
        skips: list[torch.Tensor] = []
        for i, (enc, film) in enumerate(zip(self.encoders, self.film_enc)):
            h = enc(h)              # (B, Ci, N, Ti)
            h = film(h, regime_embed)

            # 안정화: encoder stage별 activation clamp
            h = torch.clamp(h, -10.0, 10.0)

            skips.append(h)

            # Max-pool along time for all but the last encoder stage
            if i < self.n_layers - 1:
                h = F.max_pool2d(h, kernel_size=(1, 2))  # N stays, T halves

        # h is now bottleneck representation
        B, C_bot, N_bot, T_bot = h.shape

        # === Bottleneck Attention (Eq. 8 in thesis) ===
        # Numerical stability check before attention
        if torch.isnan(h).any() or torch.isinf(h).any():
            warnings.warn(
                "NaN/Inf detected before attention; applying nan_to_num",
                RuntimeWarning
            )
            h = torch.nan_to_num(h, nan=0.0, posinf=1e6, neginf=-1e6)

        # Reshape to (B*N, T_bot, C_bot) for multi-head attention
        h_flat = h.permute(0, 2, 3, 1).reshape(B * N_bot, T_bot, C_bot)
        h_attn, _ = self.bottleneck_attn(h_flat, h_flat, h_flat)
        h_attn = h_attn.reshape(B, N_bot, T_bot, C_bot).permute(0, 3, 1, 2)

        # Attention output sanity check
        if torch.isnan(h_attn).any() or torch.isinf(h_attn).any():
            warnings.warn(
                "NaN/Inf detected in attention output; zeroing attention",
                RuntimeWarning
            )
            h_attn = torch.zeros_like(h_attn)

        # Gated residual connection with clipped gate for stability
        lambda_gate = torch.sigmoid(self._lambda_gate).clamp(0.1, 0.9)
        h = h + (1.0 - lambda_gate) * h_attn

        # === Decoder with skip concatenation (Eq. 9 in thesis) ===
        # Consume skips in reverse order, except the deepest one
        dec_skips = list(reversed(skips[:-1]))

        for dec, film_dec, skip in zip(self.decoders, self.film_dec, dec_skips):
            up = dec["up"]
            conv = dec["conv"]

            # Upsample only in time dimension
            h = up(h)

            # Align spatial dimensions if mismatch due to pooling/stride
            if h.shape[-1] != skip.shape[-1] or h.shape[-2] != skip.shape[-2]:
                h = F.interpolate(
                    h,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )

            # Concatenate along channel dimension
            h = torch.cat([h, skip], dim=1)
            h = conv(h)
            h = film_dec(h, regime_embed)

            # 안정화: decoder stage별 activation clamp
            h = torch.clamp(h, -10.0, 10.0)

        # === Output head ===
        out = self.out_head(h)  # (B, 1, N, T_out)

        # 안정화: 최종 출력 clamp
        out = torch.clamp(out, -10.0, 10.0)

        out = out[:, 0]  # (B, N, T_out)

        # Reorder to (B, T_out, N) for consistency with dataloader
        out_seq = out.permute(0, 2, 1)  # (B, T_out, N)

        if return_sequence:
            return out_seq

        # === Aggregate over time ===
        if self.output_aggregation == 'last':
            # Use last time step for portfolio decision
            return out_seq[:, -1, :]

        elif self.output_aggregation == 'mean':
            # Simple average over time
            return out_seq.mean(dim=1)

        elif self.output_aggregation == 'attention':
            # Temporal attention pooling over the sequence
            B_out, T_out, N_out = out_seq.shape

            # Handle potential dimension mismatch
            if N_out != self.attn_pool.embed_dim:
                warnings.warn(
                    f"Asset count {N_out} doesn't match attn_pool embed_dim "
                    f"{self.attn_pool.embed_dim}. Using mean pooling instead.",
                    RuntimeWarning
                )
                return out_seq.mean(dim=1)

            # Self-attention over time dimension
            attn_out, _ = self.attn_pool(out_seq, out_seq, out_seq)  # (B, T_out, N)
            # Mean over time of attended representation
            return attn_out.mean(dim=1)

        else:
            raise ValueError(
                f"Unknown output_aggregation: {self.output_aggregation}"
            )


# Example usage and testing
if __name__ == "__main__":
    # Test model instantiation
    model = TimeSeriesUNet(
        input_dim=46,
        base_channels=64,
        n_layers=3,
        regime_embed_dim=32,
        n_attention_heads=4,
        dropout=0.1,
        output_aggregation='last',
    )

    print("Model initialized successfully")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    batch_size = 2
    time_steps = 60
    n_assets = 200
    n_features = 46

    x = torch.randn(batch_size, time_steps, n_assets, n_features)
    regime_embed = torch.randn(batch_size, 32)

    # Forward pass
    with torch.no_grad():
        output = model(x, regime_embed, return_sequence=False)
        print(f"\nInput shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Expected: ({batch_size}, {n_assets})")

        # Test sequence output
        output_seq = model(x, regime_embed, return_sequence=True)
        print(f"\nSequence output shape: {output_seq.shape}")
        print(f"Expected: ({batch_size}, T_out, {n_assets})")

    print("\nModel test passed!")
