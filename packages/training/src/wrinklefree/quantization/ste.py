"""Straight-Through Estimator (STE) for quantization-aware training."""

import torch


class StraightThroughEstimator(torch.autograd.Function):
    """
    Straight-Through Estimator for gradient flow through quantization.

    Forward: Apply quantization function
    Backward: Pass gradients through unchanged (identity)

    This allows training with discrete weights while maintaining gradient flow.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, quant_fn: callable) -> torch.Tensor:
        """Apply quantization in forward pass."""
        return quant_fn(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Pass gradients through unchanged."""
        return grad_output, None


def ste_quantize(x: torch.Tensor, quant_fn: callable) -> torch.Tensor:
    """
    Apply quantization with STE gradient estimation.

    Args:
        x: Input tensor to quantize
        quant_fn: Quantization function to apply

    Returns:
        Quantized tensor with gradients flowing through via STE
    """
    return StraightThroughEstimator.apply(x, quant_fn)


def detach_quantize(x: torch.Tensor, quant_fn: callable) -> torch.Tensor:
    """
    Alternative STE using detach trick.

    x_quant = x + (quant(x) - x).detach()

    This is mathematically equivalent to STE but sometimes more stable.

    Args:
        x: Input tensor to quantize
        quant_fn: Quantization function to apply

    Returns:
        Quantized tensor with gradients flowing to x
    """
    x_quant = quant_fn(x)
    return x + (x_quant - x).detach()
