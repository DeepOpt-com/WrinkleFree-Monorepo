"""Ternary weight quantization for BitNet 1.58-bit models."""

import torch


def ternary_weight_quantization(
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Quantize weights to ternary values {-1, 0, 1}.

    Uses per-tensor absmean quantization:
        scale = 1 / mean(|W|)
        W_quant = round(clip(W * scale, -1, 1)) / scale

    Args:
        weight: Weight tensor of any shape
        eps: Small constant for numerical stability

    Returns:
        Quantized weight tensor with values in {-1, 0, 1} * scale
    """
    # Compute scaling factor (absmean)
    scale = 1.0 / weight.abs().mean().clamp(min=eps)

    # Quantize: scale -> round -> clip -> unscale
    weight_scaled = weight * scale
    weight_quant = weight_scaled.round().clamp(-1, 1)

    # Unscale to preserve magnitude information
    return weight_quant / scale


def ternary_weight_quantization_no_scale(
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize weights to ternary values, returning both quantized weights and scale.

    This variant returns the raw ternary values {-1, 0, 1} and the scale separately,
    useful for inference where we want to pack weights efficiently.

    Args:
        weight: Weight tensor of any shape
        eps: Small constant for numerical stability

    Returns:
        Tuple of (quantized weights in {-1, 0, 1}, scale factor)
    """
    scale = weight.abs().mean().clamp(min=eps)
    weight_scaled = weight / scale
    weight_quant = weight_scaled.round().clamp(-1, 1)

    return weight_quant, scale


def compute_weight_scale(weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Compute the absmean scale factor for weight quantization.

    Args:
        weight: Weight tensor
        eps: Small constant for numerical stability

    Returns:
        Scale factor (scalar tensor)
    """
    return weight.abs().mean().clamp(min=eps)
