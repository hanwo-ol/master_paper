"""
TaskContextEncoder.py - Fixed Version

Key fix: Use GroupNorm instead of LayerNorm for conv blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F_func
import math
import warnings
from typing import Optional, Dict, Tuple


# ============================================================================
# Utilities
# ============================================================================

def validate_tensor(x: torch.Tensor, name: str = ""):
    """Validate tensor for NaN/Inf."""
    has_nan = torch.isnan(x).any()
    has_inf = torch.isinf(x).any()
    
    if has_nan or has_inf:
        print(f"[WARNING] {name}:")
        if has_nan:
            nan_ratio = torch.isnan(x).sum().item() / x.numel() * 100
            print(f"  NaN ratio: {nan_ratio:.2f}%")
        if has_inf:
            inf_ratio = torch.isinf(x).sum().item() / x.numel() * 100
            print(f"  Inf ratio: {inf_ratio:.2f}%")
        return False
    return True


# ============================================================================
# 1. Stable FiLM Layer
# ============================================================================

class StableFiLM2d(nn.Module):
    """Bounded Feature-wise Linear Modulation."""

    def __init__(self, embed_dim: int, num_features: int, hidden_mult: int = 2):
        super().__init__()
        hidden_dim = num_features * hidden_mult
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_features * 2),
        )
        
        # Layer normalization for context input stability
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, h: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, C, N, T = h.shape
        
        # Normalize context input
        context = self.ln(context)
        
        # MLP projection
        gamma_beta = self.mlp(context)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        
        # Apply bounds to prevent explosion
        # γ ∈ (0.5, 2.0): prevents vanishing/explosion
        # β ∈ (-0.5, 0.5): prevents large shifts
        gamma = torch.sigmoid(gamma) * 1.5 + 0.5
        beta = torch.tanh(beta) * 0.5
        
        # Reshape for broadcasting
        gamma = gamma.reshape(B, C, 1, 1)
        beta = beta.reshape(B, C, 1, 1)
        
        return gamma * h + beta


def _make_stable_conv_block(
    in_ch: int, 
    out_ch: int, 
    dropout: float = 0.0
) -> nn.Sequential:
    """Stable convolutional block with GroupNorm.
    
    Fixed: Use GroupNorm instead of LayerNorm for spatial dimensions.
    """
    # Use 8 groups for normalization (standard practice)
    num_groups = min(8, out_ch)  # Ensure out_ch is divisible by num_groups
    while out_ch % num_groups != 0:
        num_groups -= 1
    
    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.GroupNorm(num_groups=num_groups, num_channels=out_ch),
        nn.GELU(),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.GroupNorm(num_groups=num_groups, num_channels=out_ch),
        nn.GELU(),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


# ============================================================================
# 2. Chunked Attention (for large N)
# ============================================================================

class ChunkedMultiheadAttention(nn.Module):
    """Memory-efficient attention for large sequence lengths."""

    def __init__(self, embed_dim: int, num_heads: int = 4, chunk_size: int = 1000):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process attention in chunks if sequence is too long.
        
        Args:
            query: (B, T, D)
            key: (B, T, D)
            value: (B, T, D)
            
        Returns:
            output: (B, T, D)
            attn_weights: None (not returned for chunked)
        """
        T = query.shape[1]
        
        if T <= self.chunk_size:
            # Small sequence: use standard attention
            return self.attn(query, key, value, need_weights=False)
        
        # Large sequence: process in chunks
        outputs = []
        
        for i in range(0, T, self.chunk_size):
            chunk_query = query[:, i:i+self.chunk_size]
            
            # Use full key/value for context
            attn_out, _ = self.attn(
                chunk_query, key, value,
                need_weights=False
            )
            
            outputs.append(attn_out)
        
        output = torch.cat(outputs, dim=1)
        
        return output, None


# ============================================================================
# 3. Robust Task Context Encoder
# ============================================================================

class RobustTaskContextEncoder(nn.Module):
    """Numerically stable context encoder with NaN handling."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        context_dim: int = 32,
        use_returns: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.use_returns = use_returns

        feat_dim = input_dim + (1 if use_returns else 0)

        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, context_dim),
        )

    def forward(
        self,
        support_x: torch.Tensor,
        support_y: Optional[torch.Tensor] = None,
        asset_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T_s, N, n_feat = support_x.shape

        # 1. Clean NaN values
        support_x_clean = torch.nan_to_num(support_x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 2. Masked pooling
        if asset_mask is not None:
            mask_expanded = asset_mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, N, 1)
            masked_x = support_x_clean * mask_expanded
            
            # Average over valid elements
            sum_x = masked_x.sum(dim=(1, 2))  # (B, F)
            count = mask_expanded.sum(dim=(1, 2, 3), keepdim=True) + 1e-8
            feat = sum_x / count.squeeze()
        else:
            feat = support_x_clean.mean(dim=(1, 2))
        
        # 3. Clamp to prevent extreme values
        feat = torch.clamp(feat, min=-100, max=100)
        
        # 4. Return statistics (optional)
        if self.use_returns and support_y is not None:
            support_y_clean = torch.nan_to_num(support_y, nan=0.0, posinf=1.0, neginf=-1.0)
            
            if asset_mask is not None:
                mask_y = asset_mask.unsqueeze(1)
                masked_y = support_y_clean * mask_y
                sum_y = masked_y.sum(dim=(1, 2))
                count_y = mask_y.sum(dim=(1, 2)) + 1e-8
                avg_ret = sum_y / count_y
            else:
                avg_ret = support_y_clean.mean(dim=(1, 2))
            
            avg_ret = torch.clamp(avg_ret, min=-1.0, max=1.0)
            avg_ret = avg_ret.unsqueeze(-1)
            feat = torch.cat([feat, avg_ret], dim=-1)
        
        # 5. MLP projection
        context = self.mlp(feat)
        
        # 6. Final clipping
        context = torch.clamp(context, min=-10, max=10)
        
        return context


# ============================================================================
# 4. Stable U-Net
# ============================================================================

class StableTimeSeriesUNet(nn.Module):
    """Numerically stable U-Net with GroupNorm and chunked attention."""

    def __init__(
        self,
        input_dim: int = 50,
        base_channels: int = 64,
        n_layers: int = 3,
        channel_mult: Optional[list] = None,
        context_dim: int = 32,
        n_attention_heads: int = 4,
        dropout: float = 0.0,
        output_aggregation: str = 'last',
    ):
        super().__init__()

        assert n_layers >= 2
        if channel_mult is None:
            channel_mult = [2**i for i in range(n_layers)]

        self.input_dim = input_dim
        self.n_layers = n_layers
        self.output_aggregation = output_aggregation
        
        channels = [base_channels * m for m in channel_mult]
        self.channels = channels

        # Input projection with GroupNorm
        self.input_proj = nn.Sequential(
            nn.Conv2d(input_dim, channels[0], kernel_size=1),
            nn.GroupNorm(num_groups=8, num_channels=channels[0]),
        )
        
        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.LayerNorm(context_dim),
        )

        # Encoder
        self.encoders = nn.ModuleList()
        self.film_enc = nn.ModuleList()
        
        in_ch = channels[0]
        for c in channels:
            self.encoders.append(_make_stable_conv_block(in_ch, c, dropout))
            self.film_enc.append(StableFiLM2d(context_dim, c))
            in_ch = c

        # Bottleneck attention (chunked for large N)
        self.bottleneck_attn = ChunkedMultiheadAttention(
            embed_dim=channels[-1],
            num_heads=n_attention_heads,
            chunk_size=1000,
        )
        self._lambda_gate = nn.Parameter(torch.tensor(0.0))

        # Decoder
        self.decoders = nn.ModuleList()
        self.film_dec = nn.ModuleList()
        
        for i in reversed(range(n_layers - 1)):
            in_ch = channels[i + 1]
            out_ch = channels[i]
            up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=(1, 2), stride=(1, 2))
            conv = _make_stable_conv_block(out_ch * 2, out_ch, dropout)
            self.decoders.append(nn.ModuleDict({"up": up, "conv": conv}))
            self.film_dec.append(StableFiLM2d(context_dim, out_ch))

        # Output head with GroupNorm
        self.out_head = nn.Sequential(
            nn.Conv2d(channels[0], 1, kernel_size=1),
            nn.GroupNorm(num_groups=1, num_channels=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        B, T, N, n_feat = x.shape
        assert n_feat == self.input_dim

        # Input transformation
        x = x.permute(0, 3, 2, 1)  # (B, F, N, T)
        h = self.input_proj(x)
        context = self.context_proj(context)

        # Encoder
        skips = []
        for i, (enc, film) in enumerate(zip(self.encoders, self.film_enc)):
            h = enc(h)
            h = film(h, context)
            skips.append(h)

            if i < self.n_layers - 1:
                h = F_func.max_pool2d(h, kernel_size=(1, 2))

        # Bottleneck attention (stable)
        B, C_bot, N_bot, T_bot = h.shape
        h_flat = h.permute(0, 2, 3, 1).reshape(B * N_bot, T_bot, C_bot)
        
        # Layer normalization before attention for stability
        h_flat = F_func.layer_norm(h_flat, (C_bot,))
        
        h_attn, _ = self.bottleneck_attn(h_flat, h_flat, h_flat)
        h_attn = h_attn.reshape(B, N_bot, T_bot, C_bot).permute(0, 3, 1, 2)

        lambda_gate = torch.sigmoid(self._lambda_gate).clamp(0.1, 0.9)
        h = h + (1.0 - lambda_gate) * h_attn

        # Decoder
        dec_skips = list(reversed(skips[:-1]))
        
        for dec, film_dec, skip in zip(self.decoders, self.film_dec, dec_skips):
            h = dec["up"](h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F_func.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = dec["conv"](h)
            h = film_dec(h, context)

        # Output
        out = self.out_head(h)
        out = out[:, 0]
        out_seq = out.permute(0, 2, 1)

        # Clamp output
        out_seq = torch.clamp(out_seq, min=-10, max=10)

        if return_sequence:
            return out_seq
        
        if self.output_aggregation == 'last':
            return out_seq[:, -1, :]
        elif self.output_aggregation == 'mean':
            return out_seq.mean(dim=1)
        else:
            raise ValueError(f"Unknown aggregation: {self.output_aggregation}")


# ============================================================================
# 5. Stable Portfolio Backbone
# ============================================================================

class StablePortfolioBackbone(nn.Module):
    """Numerically stable portfolio backbone."""

    def __init__(
        self,
        input_dim: int,
        context_dim: int = 32,
        unet_config: Optional[Dict] = None,
        context_config: Optional[Dict] = None,
    ):
        super().__init__()

        if unet_config is None:
            unet_config = {
                'base_channels': 64,
                'n_layers': 3,
                'n_attention_heads': 4,
                'dropout': 0.1,
            }

        if context_config is None:
            context_config = {
                'hidden_dim': 128,
                'use_returns': True,
            }

        self.context_encoder = RobustTaskContextEncoder(
            input_dim=input_dim,
            hidden_dim=context_config['hidden_dim'],
            context_dim=context_dim,
            use_returns=context_config['use_returns'],
        )

        self.unet = StableTimeSeriesUNet(
            input_dim=input_dim,
            base_channels=unet_config['base_channels'],
            n_layers=unet_config['n_layers'],
            context_dim=context_dim,
            n_attention_heads=unet_config['n_attention_heads'],
            dropout=unet_config['dropout'],
        )

    def forward_support(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        asset_mask: Optional[torch.Tensor] = None,
        return_sequence: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        context = self.context_encoder(support_x, support_y, asset_mask)
        pred = self.unet(support_x, context, return_sequence)
        return pred, context

    def forward_query(
        self,
        query_x: torch.Tensor,
        context: torch.Tensor,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        return self.unet(query_x, context, return_sequence)

    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        asset_mask: Optional[torch.Tensor] = None,
        loss_type: str = "mse",
    ) -> torch.Tensor:
        assert pred.shape == target.shape

        if loss_type == "mse":
            loss_elements = (pred - target) ** 2
        elif loss_type == "mae":
            loss_elements = torch.abs(pred - target)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        if asset_mask is not None:
            if pred.dim() == 3:
                mask_expanded = asset_mask.unsqueeze(1)
            else:
                mask_expanded = asset_mask

            loss_elements = loss_elements * mask_expanded
            n_valid = mask_expanded.sum()
        else:
            n_valid = pred.numel()

        loss = loss_elements.sum() / (n_valid + 1e-8)
        
        # Clamp loss
        loss = torch.clamp(loss, min=0, max=1000)
        
        return loss
