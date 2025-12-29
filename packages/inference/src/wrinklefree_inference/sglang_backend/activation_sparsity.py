"""Activation sparsity for BitNet inference optimization.

Implements Q-Sparse-style top-k activation sparsity at inference time.
Research shows optimal sparsity of ~61.25% for 1.58-bit models.

References:
- Q-Sparse: https://arxiv.org/abs/2407.10969
- BitNet a4.8: https://arxiv.org/abs/2411.04965
- DejaVu: https://arxiv.org/abs/2310.17157
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class SparsityMode(str, Enum):
    """Activation sparsity modes."""
    NONE = "none"
    THRESHOLD = "threshold"  # Zero out activations below threshold
    TOP_K = "top_k"          # Keep top-k% activations by magnitude
    ADAPTIVE = "adaptive"    # Entropy-based adaptive sparsity


@dataclass
class ActivationSparsityConfig:
    """Configuration for activation sparsity.

    Attributes:
        enabled: Whether to apply activation sparsity
        mode: Sparsity mode (none, threshold, top_k, adaptive)
        threshold: For threshold mode, zero out activations below this value
        top_k_ratio: For top_k mode, keep this fraction of activations (0.0-1.0)
        adaptive_min_ratio: For adaptive mode, minimum keep ratio
        adaptive_max_ratio: For adaptive mode, maximum keep ratio
        track_stats: Whether to track sparsity statistics
    """
    enabled: bool = False
    mode: SparsityMode = SparsityMode.NONE
    threshold: float = 0.01
    top_k_ratio: float = 0.4  # 60% sparsity (Q-Sparse optimal for 1.58-bit)
    adaptive_min_ratio: float = 0.3
    adaptive_max_ratio: float = 0.7
    track_stats: bool = False

    # Runtime statistics (populated if track_stats=True)
    _sparsity_history: list = field(default_factory=list, repr=False)

    def get_average_sparsity(self) -> float:
        """Get average sparsity ratio from history."""
        if not self._sparsity_history:
            return 0.0
        return sum(self._sparsity_history) / len(self._sparsity_history)

    def clear_stats(self) -> None:
        """Clear sparsity statistics."""
        self._sparsity_history.clear()


def measure_sparsity(x: torch.Tensor, threshold: float = 1e-6) -> float:
    """Measure the sparsity ratio of a tensor.

    Args:
        x: Input tensor
        threshold: Values below this are considered zero

    Returns:
        Sparsity ratio (0.0 = dense, 1.0 = all zeros)
    """
    num_zeros = (x.abs() < threshold).sum().item()
    total = x.numel()
    return num_zeros / total if total > 0 else 0.0


def apply_threshold_sparsity(
    x: torch.Tensor,
    threshold: float,
) -> Tuple[torch.Tensor, float]:
    """Zero out activations below threshold. Per-token application.

    Args:
        x: Input tensor of shape (batch, features) or (batch, seq, features)
        threshold: Threshold below which activations are zeroed

    Returns:
        Tuple of (sparse_tensor, sparsity_ratio)
    """
    mask = x.abs() > threshold
    sparse_x = x * mask
    sparsity = 1.0 - mask.float().mean().item()
    return sparse_x, sparsity


def apply_top_k_sparsity(
    x: torch.Tensor,
    ratio: float,
) -> Tuple[torch.Tensor, float]:
    """Keep only top-k% activations per token by magnitude.

    Q-Sparse style inference-time sparsification.
    Optimal ratio for 1.58-bit models is ~0.4 (60% sparsity).

    Args:
        x: Input tensor of shape (..., features)
        ratio: Fraction of activations to keep (0.0-1.0)

    Returns:
        Tuple of (sparse_tensor, sparsity_ratio)
    """
    if ratio >= 1.0:
        return x, 0.0
    if ratio <= 0.0:
        return torch.zeros_like(x), 1.0

    # Get number of elements to keep per token
    k = max(1, int(x.shape[-1] * ratio))

    # Get indices of top-k by absolute value
    _, topk_idx = torch.topk(x.abs(), k, dim=-1)

    # Create mask and apply
    mask = torch.zeros_like(x)
    mask.scatter_(-1, topk_idx, 1.0)
    sparse_x = x * mask

    actual_sparsity = 1.0 - ratio
    return sparse_x, actual_sparsity


def apply_adaptive_sparsity(
    x: torch.Tensor,
    min_ratio: float = 0.3,
    max_ratio: float = 0.7,
) -> Tuple[torch.Tensor, float]:
    """Entropy-based adaptive sparsity. High variance → keep more activations.

    Inspired by DejaVu's contextual sparsity approach but uses a simple
    variance-based heuristic instead of a learned predictor.

    Args:
        x: Input tensor of shape (..., features)
        min_ratio: Minimum fraction of activations to keep
        max_ratio: Maximum fraction of activations to keep

    Returns:
        Tuple of (sparse_tensor, average_sparsity_ratio)
    """
    # Compute variance per token (last dim)
    var = x.var(dim=-1, keepdim=True)

    # Normalize variance to [0, 1] range
    var_min = var.min()
    var_max = var.max()
    if var_max - var_min < 1e-8:
        # All tokens have similar variance, use mean ratio
        keep_ratio = (min_ratio + max_ratio) / 2
        return apply_top_k_sparsity(x, keep_ratio)

    var_normalized = (var - var_min) / (var_max - var_min + 1e-8)

    # Map variance to keep ratio: high variance → keep more
    keep_ratio = min_ratio + (max_ratio - min_ratio) * var_normalized

    # Apply per-token top-k with adaptive k
    # This is a vectorized implementation
    k_per_token = (keep_ratio * x.shape[-1]).int().clamp(min=1, max=x.shape[-1])

    # For efficiency, we use the maximum k and then mask
    k_max = k_per_token.max().item()
    _, topk_idx = torch.topk(x.abs(), k_max, dim=-1)

    # Create position indices for comparison with k_per_token
    positions = torch.arange(k_max, device=x.device).unsqueeze(0)
    if x.dim() == 3:
        positions = positions.unsqueeze(0)

    # Mask: position < k_per_token means keep
    keep_mask = positions < k_per_token

    # Apply mask to topk indices
    mask = torch.zeros_like(x, dtype=torch.float32)
    # Scatter only the kept indices
    valid_idx = topk_idx.clone()
    valid_idx[~keep_mask] = 0  # Set invalid indices to 0 (will be overwritten)
    mask.scatter_(-1, valid_idx, keep_mask.float())
    mask = mask.to(x.dtype)

    sparse_x = x * mask

    # Compute actual average sparsity
    avg_keep_ratio = keep_ratio.mean().item()
    actual_sparsity = 1.0 - avg_keep_ratio

    return sparse_x, actual_sparsity


def apply_sparsity(
    x: torch.Tensor,
    config: ActivationSparsityConfig,
) -> Tuple[torch.Tensor, float]:
    """Apply activation sparsity based on config.

    Args:
        x: Input activation tensor
        config: Sparsity configuration

    Returns:
        Tuple of (sparse_tensor, sparsity_ratio)
    """
    if not config.enabled or config.mode == SparsityMode.NONE:
        return x, 0.0

    if config.mode == SparsityMode.THRESHOLD:
        sparse_x, sparsity = apply_threshold_sparsity(x, config.threshold)
    elif config.mode == SparsityMode.TOP_K:
        sparse_x, sparsity = apply_top_k_sparsity(x, config.top_k_ratio)
    elif config.mode == SparsityMode.ADAPTIVE:
        sparse_x, sparsity = apply_adaptive_sparsity(
            x, config.adaptive_min_ratio, config.adaptive_max_ratio
        )
    else:
        raise ValueError(f"Unknown sparsity mode: {config.mode}")

    # Track statistics if enabled
    if config.track_stats:
        config._sparsity_history.append(sparsity)

    return sparse_x, sparsity


# Convenience functions for common configurations

def get_default_config() -> ActivationSparsityConfig:
    """Get default sparsity config (disabled)."""
    return ActivationSparsityConfig(enabled=False)


def get_qsparse_config() -> ActivationSparsityConfig:
    """Get Q-Sparse optimal config for 1.58-bit models (60% sparsity)."""
    return ActivationSparsityConfig(
        enabled=True,
        mode=SparsityMode.TOP_K,
        top_k_ratio=0.4,  # Keep 40% → 60% sparsity
        track_stats=True,
    )


def get_conservative_config() -> ActivationSparsityConfig:
    """Get conservative sparsity config (30% sparsity)."""
    return ActivationSparsityConfig(
        enabled=True,
        mode=SparsityMode.TOP_K,
        top_k_ratio=0.7,  # Keep 70% → 30% sparsity
        track_stats=True,
    )


def get_adaptive_config() -> ActivationSparsityConfig:
    """Get adaptive sparsity config (30-70% based on input)."""
    return ActivationSparsityConfig(
        enabled=True,
        mode=SparsityMode.ADAPTIVE,
        adaptive_min_ratio=0.3,
        adaptive_max_ratio=0.7,
        track_stats=True,
    )
