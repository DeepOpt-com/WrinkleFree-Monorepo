"""Fast Walsh-Hadamard Transform for BitNet v2 H-BitLinear.

This module provides Hadamard transform via direct matrix multiplication.
Uses cached normalized Hadamard matrices for common sizes.

Optimizations:
- Direct matmul uses highly optimized cuBLAS
- Matrices cached per (size, device, dtype)
- No intermediate allocations or graph breaks

Based on BitNet v2: https://arxiv.org/abs/2504.18415
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch
import torch.nn.functional as F


def next_power_of_2(n: int) -> int:
    """
    Return smallest power of 2 >= n.

    Args:
        n: Input integer

    Returns:
        Smallest power of 2 that is >= n

    Examples:
        >>> next_power_of_2(64)
        64
        >>> next_power_of_2(65)
        128
        >>> next_power_of_2(576)
        1024
    """
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


# ============================================================================
# Hadamard Matrix Construction
# ============================================================================

# Cache for Hadamard matrices per (size, device, dtype)
_hadamard_cache: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}


def _build_hadamard_matrix(n: int) -> torch.Tensor:
    """Build n×n Hadamard matrix recursively (Sylvester construction)."""
    if n == 1:
        return torch.tensor([[1.0]])
    h_half = _build_hadamard_matrix(n // 2)
    return torch.cat([
        torch.cat([h_half, h_half], dim=1),
        torch.cat([h_half, -h_half], dim=1),
    ], dim=0)


def _get_hadamard_matrix(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Get cached normalized Hadamard matrix."""
    key = (n, device, dtype)
    if key not in _hadamard_cache:
        # Build and normalize: H @ H = n*I, so use 1/sqrt(n) for orthonormal
        h = _build_hadamard_matrix(n).to(device=device, dtype=dtype)
        h = h / math.sqrt(n)
        _hadamard_cache[key] = h
    return _hadamard_cache[key]


# ============================================================================
# Main Hadamard Transform
# ============================================================================


def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    Fast Walsh-Hadamard Transform via direct matrix multiplication.

    Uses cuBLAS for efficient GPU computation. The Hadamard matrix is cached
    per (size, device, dtype) for reuse.

    Mathematical property: H @ H = n * I (Hadamard is orthogonal up to scaling)
    For normalized Hadamard (built-in): H_norm @ H_norm = I

    Args:
        x: Input tensor of shape (..., n) where n must be a power of 2
        scale: Additional output scaling factor (default 1.0, normalization built-in)

    Returns:
        Hadamard-transformed tensor of same shape

    Raises:
        AssertionError: If last dimension is not a power of 2

    Example:
        >>> x = torch.randn(4, 32, 128)
        >>> y = hadamard_transform(x)  # Already normalized
        >>> # Apply again to invert (normalized Hadamard is involutory)
        >>> x_reconstructed = hadamard_transform(y)
        >>> torch.allclose(x, x_reconstructed, atol=1e-5)
        True
    """
    n = x.size(-1)
    assert n > 0 and (n & (n - 1)) == 0, f"Dimension must be power of 2, got {n}"

    # Get cached normalized Hadamard matrix
    H = _get_hadamard_matrix(n, x.device, x.dtype)

    # Direct matmul: x @ H (H is already normalized by 1/sqrt(n))
    result = x @ H

    # Apply additional scale if needed
    if scale != 1.0:
        result = result * scale

    return result


def hadamard_transform_weights(weight: torch.Tensor) -> torch.Tensor:
    """
    Apply Hadamard transform to weight matrix for HBitLinear conversion.

    For weight shape (out_features, in_features), applies Hadamard along
    the input dimension: W' = W @ H (each row is transformed).

    This is used during model conversion to pre-transform weights so that
    the forward pass only needs to apply Hadamard to activations.

    Mathematical equivalence:
        Y = W @ X
          = (W @ H) @ (H @ X)  # Insert H @ H = I (normalized)
          = W' @ X'            # W' = hadamard_transform_weights(W)

    Args:
        weight: Weight tensor of shape (out_features, in_features)

    Returns:
        Hadamard-transformed weight tensor of shape (out_features, padded_in)
        where padded_in is the next power of 2 >= in_features.

    Note:
        Non-power-of-2 in_features are padded to next power of 2.
        The output is NOT sliced back - this preserves H @ H = I.
    """
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D weight tensor, got shape {weight.shape}")

    out_features, in_features = weight.shape
    padded_in = next_power_of_2(in_features)

    # Pad if needed
    if padded_in != in_features:
        weight = F.pad(weight, (0, padded_in - in_features))

    # Apply normalized Hadamard to each row (input dimension is last)
    # Note: scale=1.0 since normalization is built into the matrix
    weight_h = hadamard_transform(weight, scale=1.0)

    # NO SLICING - keep full padded dimension for mathematical correctness
    # Slicing would break H @ H = I property (H_11 @ H_11 != I for submatrix)
    return weight_h
