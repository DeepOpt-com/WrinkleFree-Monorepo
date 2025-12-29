"""Adaptive Sparse Attention for BitNet inference optimization.

Implements attention sparsity techniques:
1. Top-k attention: Keep only top-k attention scores per query
2. Threshold attention: Zero out attention below threshold
3. Window attention: Local + strided global attention (Longformer-style)
4. Dynamic sparsity: Adjust based on sequence length/complexity

References:
- DejaVu: https://arxiv.org/abs/2310.17157 (contextual sparsity)
- Longformer: https://arxiv.org/abs/2004.05150 (sliding window)
- BigBird: https://arxiv.org/abs/2007.14062 (sparse patterns)
- SGLang DeepSeek-V3: Sparse attention support (2024)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class AttentionSparsityMode(str, Enum):
    """Attention sparsity modes."""
    NONE = "none"
    TOP_K = "top_k"           # Keep top-k attention scores per query
    THRESHOLD = "threshold"    # Zero out attention below threshold
    WINDOW = "window"          # Sliding window + global tokens
    DYNAMIC = "dynamic"        # Adaptive based on sequence complexity


@dataclass
class AttentionSparsityConfig:
    """Configuration for attention sparsity.

    Attributes:
        enabled: Whether to apply attention sparsity
        mode: Sparsity mode
        top_k: For top_k mode, number of keys to attend to per query
        top_k_ratio: Alternative to top_k, as ratio of sequence length
        threshold: For threshold mode, minimum attention score to keep
        window_size: For window mode, local attention window size
        global_tokens: For window mode, number of global tokens (CLS, etc.)
        stride: For window mode, stride between global attention positions
    """
    enabled: bool = False
    mode: AttentionSparsityMode = AttentionSparsityMode.NONE

    # Top-k parameters
    top_k: Optional[int] = None  # Absolute number
    top_k_ratio: float = 0.25    # Ratio of seq_len (if top_k not set)

    # Threshold parameters
    threshold: float = 0.01

    # Window parameters
    window_size: int = 256
    global_tokens: int = 1  # Usually CLS token
    stride: int = 64        # Global attention every N tokens

    # Dynamic parameters
    dynamic_min_ratio: float = 0.1
    dynamic_max_ratio: float = 0.5

    # Statistics
    track_stats: bool = False
    _sparsity_history: list = field(default_factory=list, repr=False)

    def get_average_sparsity(self) -> float:
        """Get average attention sparsity from history."""
        if not self._sparsity_history:
            return 0.0
        return sum(self._sparsity_history) / len(self._sparsity_history)


def create_window_mask(
    seq_len: int,
    window_size: int,
    global_tokens: int = 1,
    stride: int = 64,
    device: torch.device = None,
) -> torch.Tensor:
    """Create a sparse attention mask for sliding window + global attention.

    Args:
        seq_len: Sequence length
        window_size: Local window size
        global_tokens: Number of global tokens at start
        stride: Stride between global attention positions
        device: Target device

    Returns:
        Boolean mask of shape (seq_len, seq_len) where True = attend
    """
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

    for i in range(seq_len):
        # Local window attention
        start = max(0, i - window_size // 2)
        end = min(seq_len, i + window_size // 2 + 1)
        mask[i, start:end] = True

        # Global tokens (first N tokens can attend/be attended by all)
        mask[i, :global_tokens] = True
        mask[:global_tokens, i] = True

        # Strided global attention
        if i % stride == 0:
            mask[i, ::stride] = True
            mask[::stride, i] = True

    return mask


def apply_top_k_attention(
    attn_weights: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, float]:
    """Apply top-k sparsity to attention weights.

    Args:
        attn_weights: Attention weights of shape (..., seq_len, seq_len)
        k: Number of keys to keep per query

    Returns:
        Tuple of (sparse_attention_weights, sparsity_ratio)
    """
    seq_len = attn_weights.shape[-1]
    k = min(k, seq_len)

    # Get top-k indices
    _, topk_idx = torch.topk(attn_weights, k, dim=-1)

    # Create mask
    mask = torch.zeros_like(attn_weights, dtype=torch.bool)
    mask.scatter_(-1, topk_idx, True)

    # Apply mask (set non-top-k to -inf before softmax, or 0 after)
    sparse_attn = attn_weights * mask.float()

    # Renormalize (since we removed some attention)
    sparse_attn = sparse_attn / (sparse_attn.sum(dim=-1, keepdim=True) + 1e-8)

    sparsity = 1.0 - (k / seq_len)
    return sparse_attn, sparsity


def apply_threshold_attention(
    attn_weights: torch.Tensor,
    threshold: float,
) -> Tuple[torch.Tensor, float]:
    """Apply threshold-based sparsity to attention weights.

    Args:
        attn_weights: Attention weights (after softmax)
        threshold: Minimum attention weight to keep

    Returns:
        Tuple of (sparse_attention_weights, sparsity_ratio)
    """
    mask = attn_weights > threshold
    sparse_attn = attn_weights * mask.float()

    # Renormalize
    sparse_attn = sparse_attn / (sparse_attn.sum(dim=-1, keepdim=True) + 1e-8)

    sparsity = 1.0 - mask.float().mean().item()
    return sparse_attn, sparsity


def apply_window_attention(
    attn_weights: torch.Tensor,
    window_size: int,
    global_tokens: int = 1,
    stride: int = 64,
) -> Tuple[torch.Tensor, float]:
    """Apply sliding window attention sparsity.

    Args:
        attn_weights: Attention weights of shape (..., seq_len, seq_len)
        window_size: Local window size
        global_tokens: Number of global tokens
        stride: Global attention stride

    Returns:
        Tuple of (sparse_attention_weights, sparsity_ratio)
    """
    seq_len = attn_weights.shape[-1]
    device = attn_weights.device

    # Create window mask
    mask = create_window_mask(seq_len, window_size, global_tokens, stride, device)

    # Apply mask
    sparse_attn = attn_weights * mask.float()

    # Renormalize
    sparse_attn = sparse_attn / (sparse_attn.sum(dim=-1, keepdim=True) + 1e-8)

    sparsity = 1.0 - mask.float().mean().item()
    return sparse_attn, sparsity


def apply_dynamic_attention(
    attn_weights: torch.Tensor,
    min_ratio: float = 0.1,
    max_ratio: float = 0.5,
) -> Tuple[torch.Tensor, float]:
    """Apply dynamic sparsity based on attention entropy.

    Low entropy (focused attention) → more sparsity is safe
    High entropy (diffuse attention) → less sparsity

    Args:
        attn_weights: Attention weights (after softmax)
        min_ratio: Minimum fraction of tokens to keep
        max_ratio: Maximum fraction of tokens to keep

    Returns:
        Tuple of (sparse_attention_weights, sparsity_ratio)
    """
    seq_len = attn_weights.shape[-1]

    # Compute attention entropy per query
    entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-8), dim=-1)
    max_entropy = math.log(seq_len)

    # Normalize entropy to [0, 1]
    normalized_entropy = entropy / max_entropy

    # Map entropy to keep ratio: low entropy → low ratio, high → high ratio
    # (low entropy means focused attention, can be more sparse)
    keep_ratio = min_ratio + (max_ratio - min_ratio) * normalized_entropy

    # Apply per-query top-k with adaptive k
    k_per_query = (keep_ratio * seq_len).int().clamp(min=1, max=seq_len)

    # For efficiency, use max k and mask
    k_max = k_per_query.max().item()
    _, topk_idx = torch.topk(attn_weights, k_max, dim=-1)

    positions = torch.arange(k_max, device=attn_weights.device)
    keep_mask = positions.unsqueeze(0) < k_per_query.unsqueeze(-1)

    mask = torch.zeros_like(attn_weights, dtype=torch.bool)
    mask.scatter_(-1, topk_idx, keep_mask)

    sparse_attn = attn_weights * mask.float()
    sparse_attn = sparse_attn / (sparse_attn.sum(dim=-1, keepdim=True) + 1e-8)

    avg_sparsity = 1.0 - keep_ratio.mean().item()
    return sparse_attn, avg_sparsity


def apply_attention_sparsity(
    attn_weights: torch.Tensor,
    config: AttentionSparsityConfig,
) -> Tuple[torch.Tensor, float]:
    """Apply attention sparsity based on config.

    Args:
        attn_weights: Attention weights (after softmax)
        config: Sparsity configuration

    Returns:
        Tuple of (sparse_attention_weights, sparsity_ratio)
    """
    if not config.enabled or config.mode == AttentionSparsityMode.NONE:
        return attn_weights, 0.0

    seq_len = attn_weights.shape[-1]

    if config.mode == AttentionSparsityMode.TOP_K:
        k = config.top_k if config.top_k else int(seq_len * config.top_k_ratio)
        sparse_attn, sparsity = apply_top_k_attention(attn_weights, k)

    elif config.mode == AttentionSparsityMode.THRESHOLD:
        sparse_attn, sparsity = apply_threshold_attention(attn_weights, config.threshold)

    elif config.mode == AttentionSparsityMode.WINDOW:
        sparse_attn, sparsity = apply_window_attention(
            attn_weights, config.window_size, config.global_tokens, config.stride
        )

    elif config.mode == AttentionSparsityMode.DYNAMIC:
        sparse_attn, sparsity = apply_dynamic_attention(
            attn_weights, config.dynamic_min_ratio, config.dynamic_max_ratio
        )

    else:
        raise ValueError(f"Unknown attention sparsity mode: {config.mode}")

    if config.track_stats:
        config._sparsity_history.append(sparsity)

    return sparse_attn, sparsity


# Convenience functions

def get_default_attention_config() -> AttentionSparsityConfig:
    """Default config (no attention sparsity)."""
    return AttentionSparsityConfig(enabled=False)


def get_top_k_attention_config(k: int = 64) -> AttentionSparsityConfig:
    """Top-k attention config."""
    return AttentionSparsityConfig(
        enabled=True,
        mode=AttentionSparsityMode.TOP_K,
        top_k=k,
        track_stats=True,
    )


def get_window_attention_config(window_size: int = 256) -> AttentionSparsityConfig:
    """Sliding window attention config (Longformer-style)."""
    return AttentionSparsityConfig(
        enabled=True,
        mode=AttentionSparsityMode.WINDOW,
        window_size=window_size,
        global_tokens=1,
        stride=64,
        track_stats=True,
    )


def get_dynamic_attention_config() -> AttentionSparsityConfig:
    """Dynamic sparse attention based on entropy."""
    return AttentionSparsityConfig(
        enabled=True,
        mode=AttentionSparsityMode.DYNAMIC,
        dynamic_min_ratio=0.1,
        dynamic_max_ratio=0.5,
        track_stats=True,
    )
