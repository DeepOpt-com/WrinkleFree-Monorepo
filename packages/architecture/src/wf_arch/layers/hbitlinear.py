"""H-BitLinear: BitLinear with online Hadamard transformation.

Based on BitNet v2: https://arxiv.org/abs/2504.18415

H-BitLinear is used specifically for o_proj (attention output) and down_proj (FFN)
layers where activation outliers are most prominent. The Hadamard transform
decorrelates activations, enabling better low-bit quantization.

Optimizations:
- Direct matmul Hadamard uses cuBLAS (no graph breaks)
- Normalization built into cached Hadamard matrix
- BF16 consistent throughout forward pass
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from wf_arch.layers.bitlinear import BitLinear
from wf_arch.layers.hadamard import _build_hadamard_matrix, next_power_of_2
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
        3. Quantize activations: X_q = Q_8bit(X_h)
        4. Quantize weights: W_q = Q_ternary(W)
        5. Compute output: Y = W_q @ X_q

    Note: Weights are pre-transformed with Hadamard during model conversion
    using `hadamard_transform_weights()`. Both weights and activations use the
    full padded dimension to preserve H @ H = I mathematical equivalence:
        Y = W' @ X' = (W @ H) @ (H @ X) = W @ X

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
        # Compute padding for non-power-of-2 dimensions
        self.original_in_features = in_features
        padded_in = next_power_of_2(in_features)
        self.needs_padding = padded_in != in_features

        # Parent BitLinear uses padded_in as in_features for correct weight shape
        # Weight shape: (out_features, padded_in) to match Hadamard-transformed activations
        super().__init__(padded_in, out_features, bias=bias, eps=eps)

        # Pre-compute normalized Hadamard matrix as buffer (avoids graph breaks in torch.compile)
        # H @ H = n*I, so use 1/sqrt(n) for orthonormal (H_norm @ H_norm = I)
        h = _build_hadamard_matrix(padded_in) / math.sqrt(padded_in)
        self.register_buffer("hadamard_matrix", h, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with Hadamard-transformed activations.

        Uses STE (detach trick) for gradient flow through quantization.
        Supports lambda warmup for gradual quantization during training.
        All operations preserve input dtype (BF16/FP16/FP32) for consistency.

        Args:
            x: Input tensor of shape (..., original_in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        # Get lambda for gradual quantization (1.0 = full quant, 0.0 = no quant)
        lambda_val = get_current_lambda()

        # 1. Pad to power of 2 if needed (preserves dtype)
        if self.needs_padding:
            x = F.pad(x, (0, self.in_features - self.original_in_features))

        # 2. Apply Hadamard transform using pre-computed buffer (no graph breaks)
        # Cast buffer to input dtype for BF16 consistency
        H = self.hadamard_matrix.to(x.dtype)
        x_h = x @ H

        # NO SLICING - keep full padded dimension for mathematical correctness
        # Slicing would break H @ H = I property (H_11 @ H_11 != I for submatrix)
        # Weight shape is (out_features, in_features) where in_features = padded_in

        # 3. Quantize activations with STE (gradual with lambda warmup)
        x_quant = x_h + lambda_val * (self.activation_quant(x_h) - x_h).detach()

        # 4. Quantize weights with STE (cast to input dtype for BF16 consistency)
        w = self.weight.to(x.dtype)
        w_quant = w + lambda_val * (self.weight_quant(w) - w).detach()

        # 5. Linear operation
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x_quant, w_quant, bias)

    def extra_repr(self) -> str:
        return (
            f"original_in={self.original_in_features}, padded_in={self.in_features}, "
            f"out={self.out_features}, bias={self.bias is not None}"
        )
