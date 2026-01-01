"""BitLinear: Quantized linear layer for BitNet 1.58-bit models."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bitnet_arch.quantization.lambda_warmup import get_current_lambda


class BitLinear(nn.Linear):
    """
    BitLinear: 1.58-bit weight quantization with 8-bit activation quantization.

    This layer replaces nn.Linear in BitNet models. It uses:
    - Ternary weight quantization: weights are quantized to {-1, 0, 1} * scale
    - 8-bit per-token activation quantization
    - Straight-Through Estimator (STE) for gradient flow

    Weight quantization formula:
        scale = mean(|W|)
        W_quant = round(clip(W / scale, -1, 1)) * scale

    Activation quantization formula:
        gamma = max(|X|) per token
        X_quant = round(clip(127 * X / gamma, -128, 127)) * gamma / 127

    Args:
        in_features: Size of input features
        out_features: Size of output features
        bias: Whether to include bias (typically False for BitNet)
        eps: Small constant for numerical stability
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        eps: float = 1e-5,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.eps = eps

    def weight_quant(self, w: torch.Tensor) -> torch.Tensor:
        """
        Quantize weights to ternary values {-1, 0, 1}.

        Uses per-tensor absmean quantization for stability.

        Args:
            w: Weight tensor

        Returns:
            Quantized weight tensor
        """
        # Compute scale (absmean)
        scale = 1.0 / w.abs().mean().clamp(min=self.eps)

        # Quantize: scale -> round -> clip -> unscale
        w_quant = (w * scale).round().clamp(-1, 1) / scale

        return w_quant

    def activation_quant(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize activations to 8-bit per token.

        Uses per-token absmax quantization.

        Args:
            x: Activation tensor of shape (..., in_features)

        Returns:
            Quantized activation tensor
        """
        # Per-token absmax scale
        scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=self.eps)

        # Quantize
        x_quant = (x * scale).round().clamp(-128, 127) / scale

        return x_quant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with quantized weights and activations.

        Uses STE (detach trick) for gradient flow through quantization.
        Supports lambda warmup for gradual quantization during training.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        # Get lambda for gradual quantization (1.0 = full quant, 0.0 = no quant)
        lambda_val = get_current_lambda()

        # Cast weight to input dtype for mixed precision compatibility
        w = self.weight.to(x.dtype)

        # Gradual quantization with lambda warmup
        # w_quant = w + lambda * (quant(w) - w) = (1-lambda)*w + lambda*quant(w)
        w_quant = w + lambda_val * (self.weight_quant(w) - w).detach()

        # Quantize activations with STE (also with lambda warmup)
        x_quant = x + lambda_val * (self.activation_quant(x) - x).detach()

        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x_quant, w_quant, bias)


class BitLinearNoActivationQuant(nn.Linear):
    """
    BitLinear variant without activation quantization.

    Useful for ablation studies or when activation quantization
    is handled separately (e.g., fused into attention).

    Args:
        in_features: Size of input features
        out_features: Size of output features
        bias: Whether to include bias
        eps: Small constant for numerical stability
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        eps: float = 1e-5,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.eps = eps

    def weight_quant(self, w: torch.Tensor) -> torch.Tensor:
        """Quantize weights to ternary values."""
        scale = 1.0 / w.abs().mean().clamp(min=self.eps)
        return (w * scale).round().clamp(-1, 1) / scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantized weights only (with lambda warmup)."""
        lambda_val = get_current_lambda()
        w = self.weight.to(x.dtype)
        w_quant = w + lambda_val * (self.weight_quant(w) - w).detach()
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w_quant, bias)


def convert_linear_to_bitlinear(
    module: nn.Module,
    exclude_names: list[str] | None = None,
) -> nn.Module:
    """
    Convert all nn.Linear layers in a module to BitLinear.

    This is used in Stage 1 of BitDistill to convert a pre-trained model.

    Args:
        module: The module to convert
        exclude_names: List of layer names to exclude from conversion

    Returns:
        Module with Linear layers replaced by BitLinear
    """
    exclude_names = exclude_names or []

    for name, child in module.named_children():
        if name in exclude_names:
            continue

        if isinstance(child, nn.Linear) and not isinstance(child, BitLinear):
            # Create BitLinear with same configuration
            new_linear = BitLinear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias is not None,
            )
            # Copy weights
            new_linear.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                new_linear.bias.data.copy_(child.bias.data)

            setattr(module, name, new_linear)
        else:
            convert_linear_to_bitlinear(child, exclude_names)

    return module
