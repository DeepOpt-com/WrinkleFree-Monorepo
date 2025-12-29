"""CPU fallback implementations for vllm layernorm layers.

These are PyTorch-native implementations that work on CPU without CUDA.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Reference: https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.variance_size_override = var_hidden_size if var_hidden_size != hidden_size else None

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Apply RMS normalization.

        Args:
            x: Input tensor of shape [..., hidden_size]
            residual: Optional residual tensor to add before normalization

        Returns:
            Normalized tensor, or (normalized, residual) if residual provided
        """
        if residual is not None:
            x = x + residual
            residual = x

        # Compute variance
        variance_size = self.variance_size_override or x.shape[-1]
        variance = x[..., :variance_size].pow(2).mean(-1, keepdim=True)

        # Normalize
        x = x * torch.rsqrt(variance + self.variance_epsilon)

        # Apply weight
        output = x * self.weight

        if residual is not None:
            return output, residual
        return output

    def extra_repr(self) -> str:
        return f"{self.weight.shape[0]}, eps={self.variance_epsilon}"


class GemmaRMSNorm(nn.Module):
    """Gemma-style RMS normalization with +1 offset on weights.

    Gemma uses (1 + weight) * normalized instead of weight * normalized.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Apply Gemma RMS normalization."""
        if residual is not None:
            x = x + residual
            residual = x

        # Compute variance
        variance = x.pow(2).mean(-1, keepdim=True)

        # Normalize
        x = x * torch.rsqrt(variance + self.eps)

        # Apply weight with +1 offset (Gemma style)
        output = x * (1.0 + self.weight)

        if residual is not None:
            return output, residual
        return output

    def extra_repr(self) -> str:
        return f"{self.weight.shape[0]}, eps={self.eps}"


class LayerNorm(nn.LayerNorm):
    """Layer normalization wrapper for vllm compatibility."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__(hidden_size, eps=eps, elementwise_affine=elementwise_affine)
        # Handle bias separately for compatibility
        if not bias and self.bias is not None:
            self.bias = None

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is not None:
            x = x + residual
            residual = x
            output = super().forward(x)
            return output, residual
        return super().forward(x)
