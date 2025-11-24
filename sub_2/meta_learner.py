"""
meta_learner.py - Robust FOMAML-style Meta-Learner for PortfolioBackbone (v2)

Key improvements over v1:
  - Global meta gradient norm clipping (max_meta_grad_norm)
  - Optional element-wise inner gradient clipping (max_inner_grad)
  - Episode-level validity checks (skip episodes that produce NaN/Inf)
  - Soft task weighting computed only over valid episodes
  - All NaN/Inf in predictions are sanitized via torch.nan_to_num (no warnings)
"""

from typing import Dict, Tuple, Any, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from TaskContextEncoder import StablePortfolioBackbone as PortfolioBackbone


def _clone_param_state(module: nn.Module) -> Dict[str, torch.Tensor]:
    """Clone all parameters of a module into a plain dict (name -> tensor)."""
    state: Dict[str, torch.Tensor] = {}
    for name, p in module.named_parameters():
        state[name] = p.detach().clone()
    return state


def _load_param_state_inplace(module: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    """Load parameters from state dict into module in-place (no grad tracking)."""
    with torch.no_grad():
        for name, p in module.named_parameters():
            if name in state:
                p.copy_(state[name])


def _zero_grad(module: nn.Module) -> None:
    """Set .grad of all parameters in module to None."""
    for p in module.parameters():
        p.grad = None


class FOMAMLPortfolioLearner(nn.Module):
    """First-Order MAML-style meta-learner for PortfolioBackbone (robust variant).

    Args:
        backbone: PortfolioBackbone instance
        inner_lr: Learning rate for inner-loop adaptation
        num_inner_steps: Number of gradient steps in inner loop
        loss_type: 'mse' or 'mae'
        soft_weight_alpha: Temperature for soft task weighting
        max_inner_grad: Element-wise clipping for inner-loop gradients (or None)
        max_meta_grad_norm: Global gradient norm clipping for meta-update (or None)
        skip_invalid_episodes: If True, episodes with NaN/Inf in query preds/loss
            are excluded from meta gradient aggregation.
    """

    def __init__(
        self,
        backbone: PortfolioBackbone,
        inner_lr: float = 1e-3,
        num_inner_steps: int = 1,
        loss_type: str = "mse",
        soft_weight_alpha: float = 1.0,
        max_inner_grad: float | None = 1.0,
        max_meta_grad_norm: float | None = 1.0,
        skip_invalid_episodes: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
        self.loss_type = loss_type
        self.soft_weight_alpha = soft_weight_alpha
        self.max_inner_grad = max_inner_grad
        self.max_meta_grad_norm = max_meta_grad_norm
        self.skip_invalid_episodes = skip_invalid_episodes

    # ------------------------------------------------------------------
    # Inner-loop: support-based adaptation for a single episode
    # ------------------------------------------------------------------
    def _inner_adapt(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        asset_mask: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run inner-loop adaptation for a single episode (in-place update)."""
        last_support_loss: torch.Tensor | None = None
        context: torch.Tensor | None = None

        for _ in range(self.num_inner_steps):
            # Forward on support
            pred_support, context = self.backbone.forward_support(
                support_x, support_y, asset_mask, return_sequence=True
            )

            # Sanitize prediction (remove NaN/Inf)
            pred_support = torch.nan_to_num(
                pred_support, nan=0.0, posinf=1e6, neginf=-1e6
            )

            support_loss = self.backbone.compute_loss(
                pred_support,
                support_y,
                asset_mask,
                loss_type=self.loss_type,
            )

            # Optional clamp of inner loss (just to avoid crazy magnitudes)
            support_loss = torch.clamp(support_loss, 0.0, 1e6)

            _zero_grad(self.backbone)
            support_loss.backward()

            with torch.no_grad():
                for p in self.backbone.parameters():
                    if p.grad is None:
                        continue
                    g = p.grad
                    if self.max_inner_grad is not None:
                        g = torch.clamp(g, -self.max_inner_grad, self.max_inner_grad)
                    p.add_(-self.inner_lr * g)

            last_support_loss = support_loss.detach()

        assert last_support_loss is not None
        assert context is not None

        # Detach context so we don't keep the support graph
        context = context.detach()

        return last_support_loss, context

    # ------------------------------------------------------------------
    # Meta-step
    # ------------------------------------------------------------------
    def meta_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: optim.Optimizer,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Perform one meta-update step on a batch of episodes."""
        device = next(self.parameters()).device
        B = batch["support_x"].shape[0]

        support_x = batch["support_x"].to(device)
        support_y = batch["support_y"].to(device)
        query_x = batch["query_x"].to(device)
        query_y = batch["query_y"].to(device)
        asset_mask = batch["asset_mask"].to(device)

        init_state = _clone_param_state(self.backbone)

        query_losses: List[torch.Tensor] = []
        support_losses: List[torch.Tensor] = []
        valid_episode: List[bool] = []

        # ------------------ Pass 1: compute per-episode losses ------------------
        for i in range(B):
            _load_param_state_inplace(self.backbone, init_state)

            sx = support_x[i : i + 1]
            sy = support_y[i : i + 1]
            qx = query_x[i : i + 1]
            qy = query_y[i : i + 1]
            am = asset_mask[i : i + 1]

            s_loss, context = self._inner_adapt(sx, sy, am)
            support_losses.append(s_loss)

            # Query forward
            pred_query_last = self.backbone.forward_query(
                qx, context, return_sequence=False
            )

            # Sanitize prediction
            pred_query_last = torch.nan_to_num(
                pred_query_last, nan=0.0, posinf=1e6, neginf=-1e6
            )

            target_query_last = qy[:, -1, :]

            q_loss = self.backbone.compute_loss(
                pred_query_last,
                target_query_last,
                am,
                loss_type=self.loss_type,
            )

            # Clamp query loss to a reasonable range
            q_loss = torch.clamp(q_loss, 0.0, 1e6)

            # Determine validity
            is_valid = torch.isfinite(q_loss)
            valid_episode.append(bool(is_valid.item()))

            # Even if invalid, store some finite tensor for shape
            if not is_valid:
                q_loss = torch.tensor(1e6, device=device)

            query_losses.append(q_loss.detach())

        support_losses_tensor = torch.stack(support_losses)  # (B,)
        query_losses_tensor = torch.stack(query_losses)      # (B,)
        valid_mask = torch.tensor(valid_episode, dtype=torch.bool, device=device)

        # If no valid episodes, skip update to avoid NaN weights
        if valid_mask.sum() == 0:
            meta_loss = query_losses_tensor.mean()
            logs: Dict[str, Any] = {
                "meta_loss": float(meta_loss.detach().cpu().item()),
                "support_loss_mean": float(support_losses_tensor.mean().cpu().item()),
                "support_loss_std": float(
                    support_losses_tensor.std().cpu().item()
                )
                if B > 1
                else 0.0,
                "query_loss_mean": float(query_losses_tensor.mean().cpu().item()),
                "query_loss_std": float(
                    query_losses_tensor.std().cpu().item()
                )
                if B > 1
                else 0.0,
                "weights": [0.0 for _ in range(B)],
                "n_valid_episodes": 0,
            }
            # No optimizer.step() here (no reliable signal)
            return meta_loss, logs

        # Compute soft weights only over valid episodes
        valid_query_losses = query_losses_tensor[valid_mask]
        with torch.no_grad():
            weights_valid = torch.softmax(
                -self.soft_weight_alpha * valid_query_losses, dim=0
            )  # (n_valid,)

        # Expand to full-length weights for logging; invalid episodes get 0
        weights_full = torch.zeros(B, device=device)
        weights_full[valid_mask] = weights_valid

        # ------------------ Pass 2: accumulate gradients ------------------
        grad_acc: Dict[str, torch.Tensor] = {}
        total_weighted_loss_value = 0.0

        valid_idx = torch.nonzero(valid_mask, as_tuple=False).view(-1).tolist()
        for idx_pos, i in enumerate(valid_idx):
            w_i = weights_valid[idx_pos]

            _load_param_state_inplace(self.backbone, init_state)

            sx = support_x[i : i + 1]
            sy = support_y[i : i + 1]
            qx = query_x[i : i + 1]
            qy = query_y[i : i + 1]
            am = asset_mask[i : i + 1]

            _, context = self._inner_adapt(sx, sy, am)

            pred_query_last = self.backbone.forward_query(
                qx, context, return_sequence=False
            )
            pred_query_last = torch.nan_to_num(
                pred_query_last, nan=0.0, posinf=1e6, neginf=-1e6
            )

            target_query_last = qy[:, -1, :]

            q_loss = self.backbone.compute_loss(
                pred_query_last,
                target_query_last,
                am,
                loss_type=self.loss_type,
            )
            q_loss = torch.clamp(q_loss, 0.0, 1e6)

            weighted_loss = w_i * q_loss
            total_weighted_loss_value += float(weighted_loss.detach().item())

            _zero_grad(self.backbone)
            weighted_loss.backward()

            for name, p in self.backbone.named_parameters():
                if p.grad is None:
                    continue
                if name not in grad_acc:
                    grad_acc[name] = p.grad.detach().clone()
                else:
                    grad_acc[name].add_(p.grad.detach())

        # Restore initial params and assign accumulated grads
        _load_param_state_inplace(self.backbone, init_state)
        _zero_grad(self.backbone)

        for name, p in self.backbone.named_parameters():
            if name in grad_acc:
                p.grad = grad_acc[name]

        # Global gradient norm clipping
        if self.max_meta_grad_norm is not None and self.max_meta_grad_norm > 0.0:
            clip_grad_norm_(self.backbone.parameters(), max_norm=self.max_meta_grad_norm)

        optimizer.step()
        optimizer.zero_grad()

        meta_loss = torch.sum(weights_full * query_losses_tensor)

        logs: Dict[str, Any] = {
            "meta_loss": float(meta_loss.detach().cpu().item()),
            "support_loss_mean": float(support_losses_tensor.mean().cpu().item()),
            "support_loss_std": float(
                support_losses_tensor.std().cpu().item()
            )
            if B > 1
            else 0.0,
            "query_loss_mean": float(query_losses_tensor.mean().cpu().item()),
            "query_loss_std": float(
                query_losses_tensor.std().cpu().item()
            )
            if B > 1
            else 0.0,
            "weights": weights_full.detach().cpu().tolist(),
            "n_valid_episodes": int(valid_mask.sum().item()),
            "total_weighted_loss_pass2": total_weighted_loss_value,
        }

        return meta_loss, logs
