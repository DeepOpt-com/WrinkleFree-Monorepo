"""Fast Walsh-Hadamard Transform for BitNet v2 H-BitLinear.

This module provides efficient Hadamard transform implementations for:
1. Online activation transformation during forward pass
2. Offline weight transformation during model conversion

Based on BitNet v2: https://arxiv.org/abs/2504.18415
"""

from __future__ import annotations

import math

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


def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    Fast Walsh-Hadamard Transform using iterative butterfly algorithm.

    Applies the Hadamard transform to the last dimension of the input tensor.
    The transform has O(n log n) complexity and is autograd-compatible.

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

    # Iterative butterfly algorithm
    # At each stage h, we combine pairs of elements at distance h
    h = 1
    while h < n:
        # Reshape to process pairs efficiently
        # Shape: (..., n) -> (..., n//(2h), 2, h) -> butterfly -> (..., n)
        x = x.view(*x.shape[:-1], -1, 2, h)

        # Butterfly: [a, b] -> [a+b, a-b]
        a = x[..., 0, :]  # Even indices
        b = x[..., 1, :]  # Odd indices

        # In-place would be: x[..., 0, :] = a + b; x[..., 1, :] = a - b
        # But for autograd compatibility, we use out-of-place operations
        x = torch.stack([a + b, a - b], dim=-2)

        # Flatten back
        x = x.view(*x.shape[:-3], -1)

        h *= 2

    return x * scale


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
