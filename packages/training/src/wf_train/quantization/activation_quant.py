"""8-bit activation quantization for BitNet models."""

import torch


def activation_quantization_per_token(
    x: torch.Tensor,
    bits: int = 8,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Quantize activations to 8-bit using per-token absmax scaling.

    Uses per-token quantization:
        scale = 127 / max(|X|, dim=-1)
        X_quant = round(clip(X * scale, -128, 127)) / scale

    This preserves the relative magnitudes within each token's representation.

    Args:
        x: Activation tensor of shape (..., hidden_size)
        bits: Number of bits for quantization (default 8)
        eps: Small constant for numerical stability

    Returns:
        Quantized activation tensor
    """
    # Compute per-token scale (absmax along last dimension)
    max_val = 2 ** (bits - 1) - 1  # 127 for 8-bit
    min_val = -(2 ** (bits - 1))  # -128 for 8-bit

    # Per-token absmax
    absmax = x.abs().max(dim=-1, keepdim=True).values.clamp(min=eps)
    scale = max_val / absmax

    # Quantize
    x_scaled = x * scale
    x_quant = x_scaled.round().clamp(min_val, max_val)

    # Unscale
    return x_quant / scale


def activation_quantization_per_tensor(
    x: torch.Tensor,
    bits: int = 8,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Quantize activations to 8-bit using per-tensor absmax scaling.

    Uses single scale for entire tensor:
        scale = 127 / max(|X|)
        X_quant = round(clip(X * scale, -128, 127)) / scale

    Args:
        x: Activation tensor of any shape
        bits: Number of bits for quantization (default 8)
        eps: Small constant for numerical stability

    Returns:
        Quantized activation tensor
    """
    max_val = 2 ** (bits - 1) - 1
    min_val = -(2 ** (bits - 1))

    absmax = x.abs().max().clamp(min=eps)
    scale = max_val / absmax

    x_scaled = x * scale
    x_quant = x_scaled.round().clamp(min_val, max_val)

    return x_quant / scale


def activation_quantization_absmean(
    x: torch.Tensor,
    bits: int = 8,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Quantize activations using per-token absmean scaling (alternative to absmax).

    This variant uses absmean instead of absmax for potentially smoother gradients.

    Args:
        x: Activation tensor of shape (..., hidden_size)
        bits: Number of bits for quantization (default 8)
        eps: Small constant for numerical stability

    Returns:
        Quantized activation tensor
    """
    max_val = 2 ** (bits - 1) - 1
    min_val = -(2 ** (bits - 1))

    # Per-token absmean
    absmean = x.abs().mean(dim=-1, keepdim=True).clamp(min=eps)
    # Use a scaling factor based on absmean (empirically ~2.5x absmean covers most values)
    scale = max_val / (2.5 * absmean)

    x_scaled = x * scale
    x_quant = x_scaled.round().clamp(min_val, max_val)

    return x_quant / scale
