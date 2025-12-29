"""Activation sparsification for Q-Sparse.

Implements Top-K and Block (N:M) sparsity for activations with STE gradient flow.
Reference: arxiv.org/abs/2407.10969

The key insight from Q-Sparse is that sparse activations can be combined with
quantization for multiplicative efficiency gains. For 1-bit models like BitNet b1.58,
the optimal sparsity ratio is approximately 61% (keeping 39% of activations).
"""

from __future__ import annotations

import torch


class TopKSparsity(torch.autograd.Function):
    """Top-K sparsification with STE for gradient flow.

    This custom autograd function applies sparsification in the forward pass
    while allowing gradients to flow through unchanged in the backward pass
    (Straight-Through Estimator).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, k: int) -> torch.Tensor:
        """Apply top-k sparsification along the last dimension.

        Args:
            x: Input tensor of shape (..., D)
            k: Number of elements to keep per vector

        Returns:
            Sparsified tensor with same shape as input
        """
        _, topk_idx = x.abs().topk(k, dim=-1)
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask.scatter_(-1, topk_idx, True)
        ctx.save_for_backward(mask)
        return x * mask

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """STE: pass gradients through unchanged."""
        # Gradients flow to ALL elements, not just top-k
        return grad_output, None


def topk_sparsify(
    x: torch.Tensor,
    sparsity_ratio: float,
    per_token: bool = True,
) -> torch.Tensor:
    """Apply Top-K sparsification to activations.

    Keeps the top (1 - sparsity_ratio) fraction of elements by absolute value.
    Uses STE so gradients flow through the sparsity mask unchanged.

    Args:
        x: Activation tensor of shape (..., D)
        sparsity_ratio: Fraction of elements to zero out (0.0 = dense, 1.0 = all zeros)
        per_token: If True, apply per-token (per last dim); if False, per-tensor

    Returns:
        Sparsified tensor with STE gradient flow
    """
    if sparsity_ratio <= 0:
        return x
    if sparsity_ratio >= 1:
        return torch.zeros_like(x)

    dim = x.shape[-1]
    k = max(1, int((1 - sparsity_ratio) * dim))

    if per_token:
        return TopKSparsity.apply(x, k)
    else:
        # Per-tensor: flatten, apply, reshape
        original_shape = x.shape
        x_flat = x.view(-1)
        k_total = max(1, int((1 - sparsity_ratio) * x_flat.numel()))
        _, topk_idx = x_flat.abs().topk(k_total)
        mask = torch.zeros_like(x_flat, dtype=torch.bool)
        mask.scatter_(0, topk_idx, True)
        # Use STE for per-tensor as well
        x_sparse = x_flat * mask
        return (x_flat + (x_sparse - x_flat).detach()).view(original_shape)


def block_sparsify_nm(
    x: torch.Tensor,
    n: int,
    m: int,
) -> torch.Tensor:
    """Apply N:M structured sparsity (Block Q-Sparse).

    For every M consecutive elements, keep the top N by magnitude.
    This creates hardware-friendly structured sparsity patterns.

    Args:
        x: Activation tensor of shape (..., D) where D should be divisible by M
        n: Number of elements to keep per block (N in N:M pattern)
        m: Block size (M in N:M pattern)

    Returns:
        Block-sparsified tensor with STE gradient flow
    """
    if n >= m:
        return x  # No sparsification needed

    *batch_dims, d = x.shape

    # Pad if necessary to make divisible by M
    pad_size = 0
    if d % m != 0:
        pad_size = m - (d % m)
        x = torch.nn.functional.pad(x, (0, pad_size))
        d = x.shape[-1]

    # Reshape to (..., num_blocks, m)
    x_blocked = x.view(*batch_dims, d // m, m)

    # Find top-n in each block
    _, topk_idx = x_blocked.abs().topk(n, dim=-1)
    mask = torch.zeros_like(x_blocked, dtype=torch.bool)
    mask.scatter_(-1, topk_idx, True)

    # Apply mask
    x_sparse = x_blocked * mask

    # Apply STE: gradients flow through unchanged
    x_ste = x_blocked + (x_sparse - x_blocked).detach()

    # Reshape back
    x_out = x_ste.view(*batch_dims, d)

    # Remove padding if applied
    if pad_size > 0:
        x_out = x_out[..., :-pad_size]

    return x_out


def detach_sparsify(
    x: torch.Tensor,
    sparsity_ratio: float,
    per_token: bool = True,
) -> torch.Tensor:
    """Apply sparsification with detach trick STE.

    This is an alternative STE implementation using the detach trick:
        x_sparse = x + (sparse(x) - x).detach()

    This matches the pattern used in BitLinear for quantization.

    Args:
        x: Activation tensor of shape (..., D)
        sparsity_ratio: Fraction of elements to zero out
        per_token: If True, apply per-token; if False, per-tensor

    Returns:
        Sparsified tensor with STE gradient flow
    """
    if sparsity_ratio <= 0:
        return x
    if sparsity_ratio >= 1:
        return torch.zeros_like(x)

    dim = x.shape[-1]
    k = max(1, int((1 - sparsity_ratio) * dim))

    if per_token:
        _, topk_idx = x.abs().topk(k, dim=-1)
        mask = torch.zeros_like(x)
        mask.scatter_(-1, topk_idx, 1.0)
        x_sparse = x * mask
    else:
        # Per-tensor
        original_shape = x.shape
        x_flat = x.view(-1)
        k_total = max(1, int((1 - sparsity_ratio) * x_flat.numel()))
        _, topk_idx = x_flat.abs().topk(k_total)
        mask = torch.zeros_like(x_flat)
        mask.scatter_(0, topk_idx, 1.0)
        x_sparse = (x_flat * mask).view(original_shape)

    # STE with detach trick: gradients flow to x, not through sparsity
    return x + (x_sparse - x).detach()
