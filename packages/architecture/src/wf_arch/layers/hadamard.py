"""Fast Walsh-Hadamard Transform for BitNet v2 H-BitLinear.

This module provides efficient Hadamard transform implementations for:
1. Online activation transformation during forward pass
2. Offline weight transformation during model conversion

Optimizations:
- Unrolled butterfly for common sizes (1024, 2048, 4096) for torch.compile fusion
- Optional sgl_kernel CUDA backend with fallback
- torch.compile with reduce-overhead mode for fused kernels

Based on BitNet v2: https://arxiv.org/abs/2504.18415
"""

from __future__ import annotations

import math
from typing import Callable

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
# CUDA Backend Detection (sgl_kernel)
# ============================================================================

_HAS_SGL_KERNEL = False
_sgl_hadamard: Callable | None = None

try:
    # Try to import the CUDA kernel from sgl_kernel (inference package)
    import torch.ops.sgl_kernel  # type: ignore

    _HAS_SGL_KERNEL = hasattr(torch.ops.sgl_kernel, "fast_hadamard_transform")
    if _HAS_SGL_KERNEL:
        _sgl_hadamard = torch.ops.sgl_kernel.fast_hadamard_transform.default
except (ImportError, AttributeError):
    pass


# ============================================================================
# Unrolled Hadamard for Common Sizes (torch.compile friendly)
# ============================================================================


def _butterfly_step(x: torch.Tensor, h: int) -> torch.Tensor:
    """Single butterfly step: reshape, add/sub, flatten."""
    x = x.view(*x.shape[:-1], -1, 2, h)
    a = x[..., 0, :]
    b = x[..., 1, :]
    # Use contiguous to help torch.compile fuse operations
    x = torch.stack([a + b, a - b], dim=-2)
    return x.view(*x.shape[:-3], -1)


@torch.compile(mode="reduce-overhead", disable=not torch.cuda.is_available())
def _hadamard_1024(x: torch.Tensor) -> torch.Tensor:
    """Unrolled Hadamard for 1024-dim (Qwen3-0.6B hidden size).

    10 explicit butterfly stages for torch.compile fusion.
    """
    x = _butterfly_step(x, 1)
    x = _butterfly_step(x, 2)
    x = _butterfly_step(x, 4)
    x = _butterfly_step(x, 8)
    x = _butterfly_step(x, 16)
    x = _butterfly_step(x, 32)
    x = _butterfly_step(x, 64)
    x = _butterfly_step(x, 128)
    x = _butterfly_step(x, 256)
    x = _butterfly_step(x, 512)
    return x


@torch.compile(mode="reduce-overhead", disable=not torch.cuda.is_available())
def _hadamard_2048(x: torch.Tensor) -> torch.Tensor:
    """Unrolled Hadamard for 2048-dim (common hidden size).

    11 explicit butterfly stages for torch.compile fusion.
    """
    x = _butterfly_step(x, 1)
    x = _butterfly_step(x, 2)
    x = _butterfly_step(x, 4)
    x = _butterfly_step(x, 8)
    x = _butterfly_step(x, 16)
    x = _butterfly_step(x, 32)
    x = _butterfly_step(x, 64)
    x = _butterfly_step(x, 128)
    x = _butterfly_step(x, 256)
    x = _butterfly_step(x, 512)
    x = _butterfly_step(x, 1024)
    return x


@torch.compile(mode="reduce-overhead", disable=not torch.cuda.is_available())
def _hadamard_4096(x: torch.Tensor) -> torch.Tensor:
    """Unrolled Hadamard for 4096-dim (Llama 7B hidden size).

    12 explicit butterfly stages for torch.compile fusion.
    """
    x = _butterfly_step(x, 1)
    x = _butterfly_step(x, 2)
    x = _butterfly_step(x, 4)
    x = _butterfly_step(x, 8)
    x = _butterfly_step(x, 16)
    x = _butterfly_step(x, 32)
    x = _butterfly_step(x, 64)
    x = _butterfly_step(x, 128)
    x = _butterfly_step(x, 256)
    x = _butterfly_step(x, 512)
    x = _butterfly_step(x, 1024)
    x = _butterfly_step(x, 2048)
    return x


# ============================================================================
# Main Hadamard Transform
# ============================================================================


def _hadamard_generic(x: torch.Tensor) -> torch.Tensor:
    """Generic Hadamard using iterative butterfly (fallback for uncommon sizes)."""
    n = x.size(-1)
    h = 1
    while h < n:
        x = _butterfly_step(x, h)
        h *= 2
    return x


def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    Fast Walsh-Hadamard Transform using iterative butterfly algorithm.

    Applies the Hadamard transform to the last dimension of the input tensor.
    The transform has O(n log n) complexity and is autograd-compatible.

    Optimizations:
    - Uses CUDA sgl_kernel if available (fastest)
    - Uses unrolled torch.compiled versions for 1024, 2048, 4096 dims
    - Falls back to generic loop for other sizes

    Mathematical property: H @ H = n * I (Hadamard is orthogonal up to scaling)
    For normalized Hadamard (scale=1/sqrt(n)): H_norm @ H_norm = I

    Args:
        x: Input tensor of shape (..., n) where n must be a power of 2
        scale: Output scaling factor. Use 1/sqrt(n) for normalized transform.

    Returns:
        Hadamard-transformed tensor of same shape

    Raises:
        AssertionError: If last dimension is not a power of 2

    Example:
        >>> x = torch.randn(4, 32, 128)
        >>> y = hadamard_transform(x, scale=1/math.sqrt(128))
        >>> # Apply again to invert (normalized Hadamard is involutory)
        >>> x_reconstructed = hadamard_transform(y, scale=1/math.sqrt(128))
        >>> torch.allclose(x, x_reconstructed, atol=1e-5)
        True
    """
    n = x.size(-1)
    assert n > 0 and (n & (n - 1)) == 0, f"Dimension must be power of 2, got {n}"

    # Try CUDA kernel first (fastest)
    if _HAS_SGL_KERNEL and x.is_cuda and _sgl_hadamard is not None:
        return _sgl_hadamard(x, scale)

    # Use unrolled versions for common sizes (torch.compile optimized)
    if n == 1024:
        result = _hadamard_1024(x)
    elif n == 2048:
        result = _hadamard_2048(x)
    elif n == 4096:
        result = _hadamard_4096(x)
    else:
        # Generic fallback for other sizes
        result = _hadamard_generic(x)

    # Apply scale (in-place if possible for memory efficiency)
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
          = (W @ H) @ (H @ X)  # Insert H @ H = n*I (with proper scaling)
          = W' @ X'            # W' = hadamard_transform_weights(W)

    Args:
        weight: Weight tensor of shape (out_features, in_features)

    Returns:
        Hadamard-transformed weight tensor of same shape

    Note:
        Non-power-of-2 in_features are automatically padded and sliced back.
    """
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D weight tensor, got shape {weight.shape}")

    out_features, in_features = weight.shape
    padded_in = next_power_of_2(in_features)

    # Pad if needed
    if padded_in != in_features:
        weight = F.pad(weight, (0, padded_in - in_features))

    # Apply normalized Hadamard to each row (input dimension is last)
    scale = 1.0 / math.sqrt(padded_in)
    weight_h = hadamard_transform(weight, scale=scale)

    # Slice back to original size
    if padded_in != in_features:
        weight_h = weight_h[:, :in_features]

    return weight_h
