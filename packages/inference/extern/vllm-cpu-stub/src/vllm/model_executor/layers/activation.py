"""CPU fallback implementations for vllm activation layers.

These are PyTorch-native implementations that work on CPU without CUDA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    """SiLU activation with gating: silu(x[:d]) * x[d:]."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]


class GeluAndMul(nn.Module):
    """GELU activation with gating: gelu(x[:d]) * x[d:]."""

    def __init__(self, approximate: str = "none"):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        if self.approximate == "tanh":
            return F.gelu(x[..., :d], approximate="tanh") * x[..., d:]
        return F.gelu(x[..., :d]) * x[..., d:]


class NewGELU(nn.Module):
    """GELU activation with tanh approximation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate="tanh")


class FastGELU(nn.Module):
    """Fast GELU approximation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


class QuickGELU(nn.Module):
    """Quick GELU approximation using sigmoid."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ScaledActivation(nn.Module):
    """Scaled activation function."""

    def __init__(self, act_module: nn.Module, intermediate_size: int, input_is_parallel: bool = True):
        super().__init__()
        self.act = act_module
        self.scale = nn.Parameter(torch.ones(intermediate_size))
        self.input_is_parallel = input_is_parallel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x) * self.scale
