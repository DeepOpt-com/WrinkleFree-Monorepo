"""BitNet Feed-Forward Network with SubLN and ReLU^2."""

import torch
import torch.nn as nn

from wrinklefree.models.bitlinear import BitLinear
from wrinklefree.models.subln import SubLN


class BitNetFFN(nn.Module):
    """
    BitNet Feed-Forward Network with SubLN.

    Architecture follows LLaMA-style SwiGLU but with ReLU^2 activation
    as specified in the BitNet paper:

        FFN(x) = W_down(SubLN(W_up(x) * ReLU^2(W_gate(x))))

    Where:
    - W_gate, W_up: Project from hidden_size to intermediate_size
    - ReLU^2: Squared ReLU activation
    - SubLN: Sub-layer normalization before down projection
    - W_down: Project from intermediate_size back to hidden_size

    Args:
        hidden_size: Model hidden dimension
        intermediate_size: FFN intermediate dimension (typically 4x hidden_size)
        hidden_act: Activation function ("relu2" or "silu")
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "relu2",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act

        # Projections (all BitLinear)
        self.gate_proj = BitLinear(hidden_size, intermediate_size, bias=False)
        self.up_proj = BitLinear(hidden_size, intermediate_size, bias=False)
        self.down_proj = BitLinear(intermediate_size, hidden_size, bias=False)

        # SubLN before down projection (key BitDistill modification)
        self.subln = SubLN(intermediate_size)

        # Activation function
        if hidden_act == "relu2":
            self.act_fn = self._relu_squared
        elif hidden_act == "silu":
            self.act_fn = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {hidden_act}")

    @staticmethod
    def _relu_squared(x: torch.Tensor) -> torch.Tensor:
        """ReLU^2 activation: max(0, x)^2"""
        return torch.relu(x).pow(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through FFN.

        Args:
            x: Input tensor of shape (..., hidden_size)

        Returns:
            Output tensor of shape (..., hidden_size)
        """
        # Gate and up projections
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        # Apply gated activation
        if self.hidden_act == "relu2":
            activated = up * self._relu_squared(gate)
        else:
            activated = up * self.act_fn(gate)

        # SubLN before down projection
        activated = self.subln(activated)

        # Down projection
        return self.down_proj(activated)


class BitNetMLP(nn.Module):
    """
    Simple BitNet MLP (non-gated variant).

    Architecture:
        MLP(x) = W_down(SubLN(act(W_up(x))))

    This is a simpler variant without the gating mechanism.

    Args:
        hidden_size: Model hidden dimension
        intermediate_size: FFN intermediate dimension
        hidden_act: Activation function ("relu2", "relu", "gelu")
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "relu2",
    ):
        super().__init__()
        self.up_proj = BitLinear(hidden_size, intermediate_size, bias=False)
        self.down_proj = BitLinear(intermediate_size, hidden_size, bias=False)
        self.subln = SubLN(intermediate_size)

        if hidden_act == "relu2":
            self.act_fn = lambda x: torch.relu(x).pow(2)
        elif hidden_act == "relu":
            self.act_fn = torch.relu
        elif hidden_act == "gelu":
            self.act_fn = nn.functional.gelu
        else:
            raise ValueError(f"Unknown activation: {hidden_act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through simple MLP."""
        x = self.up_proj(x)
        x = self.act_fn(x)
        x = self.subln(x)
        x = self.down_proj(x)
        return x
