"""SubLN (Sub-Layer Normalization) module for BitNet models."""

import torch
import torch.nn as nn


class SubLN(nn.Module):
    """
    Sub-Layer Normalization (SubLN) for BitNet.

    SubLN is placed before output projections in:
    - Multi-Head Self-Attention (before W_out)
    - Feed-Forward Network (before W_down)

    This stabilizes activation variance during quantized training, which is
    critical for successful BitNet training as described in the BitDistill paper.

    Uses RMSNorm variant: x / sqrt(mean(x^2) + eps) * weight

    Args:
        hidden_size: Dimension of the input features
        eps: Small constant for numerical stability
        elementwise_affine: Whether to include learnable scale parameter
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to input tensor.

        Args:
            x: Input tensor of shape (..., hidden_size)

        Returns:
            Normalized tensor of same shape
        """
        # RMSNorm: x / sqrt(mean(x^2) + eps)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)

        if self.weight is not None:
            return x_normed * self.weight
        return x_normed

    def extra_repr(self) -> str:
        return f"{self.hidden_size}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    This is the standard RMSNorm used for main layer normalization in LLaMA-style
    models. SubLN is a specific application of RMSNorm before output projections.

    Args:
        hidden_size: Dimension of the input features
        eps: Small constant for numerical stability
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to input tensor.

        Args:
            x: Input tensor of shape (..., hidden_size)

        Returns:
            Normalized tensor of same shape
        """
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return x_normed * self.weight

    def extra_repr(self) -> str:
        return f"{self.hidden_size}, eps={self.eps}"
