"""H-BitLinear: BitLinear with online Hadamard transformation.

Based on BitNet v2: https://arxiv.org/abs/2504.18415

H-BitLinear is used specifically for o_proj (attention output) and down_proj (FFN)
layers where activation outliers are most prominent. The Hadamard transform
decorrelates activations, enabling better low-bit quantization.

Optimizations:
- Pre-computed scale as buffer (correct dtype/device)
- torch.compile compatible via unrolled Hadamard
- BF16 consistent throughout forward pass
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from wf_arch.layers.bitlinear import BitLinear
from wf_arch.layers.hadamard import hadamard_transform, next_power_of_2
from wf_arch.quantization.lambda_warmup import get_current_lambda


class HBitLinear(BitLinear):
    """
    H-BitLinear: BitLinear with online Hadamard transformation.

    This layer extends BitLinear by applying the Hadamard transform to activations
    before quantization. This decorrelates activation values and reduces outliers,
    enabling better low-bit quantization as described in the BitNet v2 paper.

    Forward pass:
        1. Pad activations to next power of 2 (if needed)
        2. Apply Hadamard transform: X_h = H @ X (normalized)
        3. Slice back to original dimension
        4. Quantize activations: X_q = Q_8bit(X_h)
        5. Quantize weights: W_q = Q_ternary(W)
        6. Compute output: Y = W_q @ X_q

    Note: Weights should be pre-transformed with Hadamard during model conversion
    using `hadamard_transform_weights()`. This makes the full computation:
        Y = W' @ X' = (W @ H) @ (H @ X) = W @ X (mathematically equivalent)

    Used for:
        - o_proj in attention (W_o)
        - down_proj in FFN (W_down)

    Not used for:
        - q_proj, k_proj, v_proj (use standard BitLinear)
        - gate_proj, up_proj (use standard BitLinear)

    Args:
        in_features: Size of input features (will pad if not power of 2)
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
        super().__init__(in_features, out_features, bias=bias, eps=eps)

        # Compute padding for non-power-of-2 dimensions
        self.padded_in = next_power_of_2(in_features)
        self.needs_padding = self.padded_in != in_features

        # Register hadamard_scale as buffer for correct dtype/device handling
        # This avoids Python float -> tensor conversion in forward pass
        hadamard_scale = 1.0 / math.sqrt(self.padded_in)
        self.register_buffer(
            "hadamard_scale",
            torch.tensor(hadamard_scale, dtype=torch.float32),
            persistent=False,  # Don't save in state_dict
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with Hadamard-transformed activations.

        Uses STE (detach trick) for gradient flow through quantization.
        Supports lambda warmup for gradual quantization during training.
        All operations preserve input dtype (BF16/FP16/FP32) for consistency.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        # Get lambda for gradual quantization (1.0 = full quant, 0.0 = no quant)
        lambda_val = get_current_lambda()

        # 1. Pad to power of 2 if needed (preserves dtype)
        if self.needs_padding:
            x = F.pad(x, (0, self.padded_in - self.in_features))

        # 2. Apply Hadamard transform (decorrelates activations, reduces outliers)
        # Scale is cast to input dtype for BF16 consistency
        scale = self.hadamard_scale.to(x.dtype).item()
        x_h = hadamard_transform(x, scale=scale)

        # 3. Slice back to original dimension
        if self.needs_padding:
            x_h = x_h[..., : self.in_features]

        # 4. Quantize activations with STE (gradual with lambda warmup)
        x_quant = x_h + lambda_val * (self.activation_quant(x_h) - x_h).detach()

        # 5. Quantize weights with STE (cast to input dtype for BF16 consistency)
        w = self.weight.to(x.dtype)
        w_quant = w + lambda_val * (self.weight_quant(w) - w).detach()

        # 6. Linear operation
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x_quant, w_quant, bias)

    def extra_repr(self) -> str:
        return (
            f"{self.in_features}, {self.out_features}, "
            f"bias={self.bias is not None}, "
            f"padded_in={self.padded_in}, needs_padding={self.needs_padding}"
        )
