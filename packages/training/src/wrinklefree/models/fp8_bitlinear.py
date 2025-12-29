"""FP8-accelerated BitLinear layer using TorchAO (DeepSeek-V3 style).

This module provides BitLinear variants that use FP8 for the underlying GEMM
computation while preserving the existing INT8 activation simulation and
ternary weight quantization semantics.

The key insight is that FP8 accelerates the GEMM, not the quantization:
- BitLinear's INT8 activation quantization (STE) happens in BF16
- BitLinear's ternary weight quantization (STE) happens in BF16
- Only the final F.linear() call is replaced with FP8 GEMM

References:
- DeepSeek-V3 Technical Report: https://arxiv.org/abs/2412.19437
- TorchAO torch._scaled_mm: https://docs.pytorch.org/ao/stable/pretraining.html
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from wrinklefree.models.bitlinear import BitLinear
from wrinklefree.quantization.lambda_warmup import get_current_lambda

if TYPE_CHECKING:
    from wrinklefree.quantization.fp8_gemm import FP8Config

logger = logging.getLogger(__name__)

# Lazy import flag for TorchAO
_TORCHAO_CHECKED = False
_TORCHAO_AVAILABLE = False


def _check_torchao() -> bool:
    """Lazy check for TorchAO FP8 availability."""
    global _TORCHAO_CHECKED, _TORCHAO_AVAILABLE
    if not _TORCHAO_CHECKED:
        try:
            # Check for scaled_mm support (PyTorch 2.4+)
            if hasattr(torch, "_scaled_mm"):
                _TORCHAO_AVAILABLE = True
            else:
                logger.warning("torch._scaled_mm not available (requires PyTorch 2.4+)")
                _TORCHAO_AVAILABLE = False
        except Exception as e:
            logger.warning(f"Error checking TorchAO availability: {e}")
            _TORCHAO_AVAILABLE = False
        _TORCHAO_CHECKED = True
    return _TORCHAO_AVAILABLE


class FP8BitLinear(BitLinear):
    """BitLinear with FP8 GEMM acceleration via torch._scaled_mm.

    This layer maintains BitLinear's INT8 activation simulation and ternary
    weight quantization semantics, but uses FP8 for the underlying GEMM
    computation when hardware supports it (H100+).

    The INT8 simulation (for gradient flow via STE) happens BEFORE the GEMM.
    The FP8 is purely a hardware acceleration of the GEMM itself.

    Flow:
        1. x (BF16) -> activation_quant (INT8 sim with STE) -> x_quant (BF16)
        2. w (BF16) -> weight_quant (ternary sim with STE) -> w_quant (BF16)
        3. x_quant, w_quant -> FP8 GEMM (torch._scaled_mm) -> output (BF16)

    Args:
        in_features: Size of input features
        out_features: Size of output features
        bias: Whether to include bias
        eps: Small constant for numerical stability
        fp8_config: FP8 configuration (None = use defaults)
        layer_name: Layer name for exclusion pattern matching
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        eps: float = 1e-5,
        fp8_config: Optional["FP8Config"] = None,
        layer_name: str = "",
    ):
        super().__init__(in_features, out_features, bias=bias, eps=eps)

        # Import here to avoid circular imports
        from wrinklefree.quantization.fp8_gemm import FP8Config

        self.fp8_config = fp8_config or FP8Config()
        self.layer_name = layer_name

        # Determine if we should use FP8 for this layer
        self._use_fp8 = self._should_use_fp8()

        if self._use_fp8:
            logger.debug(f"FP8 enabled for layer: {layer_name}")

    def _should_use_fp8(self) -> bool:
        """Check if FP8 should be used for this layer."""
        if not _check_torchao():
            return False

        from wrinklefree.quantization.fp8_gemm import should_use_fp8_for_layer

        return should_use_fp8_for_layer(
            self.layer_name,
            self.in_features,
            self.out_features,
            self.fp8_config,
        )

    def _compute_fp8_scales(
        self, x: torch.Tensor, w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-row/per-column scales for FP8 quantization.

        Following DeepSeek-V3's fine-grained quantization:
        - Activations: tile-wise (1x128) -> per-row scaling
        - Weights: block-wise -> per-column scaling

        Args:
            x: Activation tensor of shape (batch*seq, in_features)
            w: Weight tensor of shape (out_features, in_features)

        Returns:
            Tuple of (x_scale, w_scale) for FP8 conversion
        """
        # E4M3 max value is ~448, we use a smaller range for safety
        FP8_MAX = 448.0

        # Per-row scale for activations (rowwise scaling)
        x_amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        x_scale = FP8_MAX / x_amax

        # Per-row scale for weights (which becomes per-column after transpose)
        w_amax = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        w_scale = FP8_MAX / w_amax

        return x_scale.squeeze(-1), w_scale.squeeze(-1)

    def _fp8_matmul(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Perform FP8-accelerated matrix multiplication.

        The inputs x and w are already quantized (INT8 sim / ternary sim in BF16).
        This function handles the FP8 casting and GEMM using torch._scaled_mm.

        Args:
            x: Quantized activation tensor of shape (..., in_features)
            w: Quantized weight tensor of shape (out_features, in_features)
            bias: Optional bias tensor

        Returns:
            Output tensor of shape (..., out_features)
        """
        # Get accumulator dtype from config
        from wrinklefree.quantization.fp8_gemm import get_accumulator_dtype

        acc_dtype = get_accumulator_dtype(self.fp8_config)

        # Remember original shape for reshaping later
        original_shape = x.shape[:-1]
        x_2d = x.view(-1, x.size(-1))  # (batch*seq, in_features)

        # Compute scales for FP8
        x_scale, w_scale = self._compute_fp8_scales(x_2d, w)

        # Quantize to FP8 E4M3 format
        # Note: We scale, clamp, and cast to FP8
        x_fp8 = (x_2d * x_scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
        w_fp8 = (w * w_scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)

        # Perform FP8 GEMM with scaled_mm
        # Note: w needs to be transposed and contiguous for scaled_mm
        # Output scale combines both input scales for proper dequantization
        output = torch._scaled_mm(
            x_fp8,
            w_fp8.t().contiguous(),
            out_dtype=acc_dtype,
            scale_a=x_scale.unsqueeze(-1).reciprocal(),
            scale_b=w_scale.unsqueeze(0).reciprocal(),
        )

        # Reshape back to original shape
        output = output.view(*original_shape, -1)

        # Add bias if present
        if bias is not None:
            output = output + bias

        # Cast back to input dtype (BF16)
        return output.to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with FP8-accelerated GEMM.

        INT8 activation simulation and ternary weight quantization happen
        BEFORE the GEMM, maintaining STE gradient flow.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        # Get lambda for quantization warmup
        lambda_val = get_current_lambda()

        # Cast weight to input dtype
        w = self.weight.to(x.dtype)

        # Apply quantization with STE (unchanged from BitLinear)
        w_quant = w + lambda_val * (self.weight_quant(w) - w).detach()
        x_quant = x + lambda_val * (self.activation_quant(x) - x).detach()

        bias = self.bias.to(x.dtype) if self.bias is not None else None

        # Use FP8 GEMM if enabled, hardware supports it, and we're training
        # (FP8 training has overhead, so we only use it when training)
        if self._use_fp8 and self.training:
            try:
                return self._fp8_matmul(x_quant, w_quant, bias)
            except Exception as e:
                # Fallback to BF16 if FP8 fails (e.g., shape requirements)
                logger.warning(f"FP8 GEMM failed for {self.layer_name}, falling back to BF16: {e}")
                return F.linear(x_quant, w_quant, bias)
        else:
            # Fallback to standard BF16 GEMM (inference or no FP8 support)
            return F.linear(x_quant, w_quant, bias)


def convert_bitlinear_to_fp8(
    module: nn.Module,
    fp8_config: Optional["FP8Config"] = None,
    prefix: str = "",
) -> nn.Module:
    """Convert BitLinear layers to FP8BitLinear for accelerated training.

    This function recursively traverses the module tree and replaces
    BitLinear layers with FP8BitLinear, preserving weights and configuration.

    Args:
        module: Module to convert
        fp8_config: FP8 configuration (None = use defaults)
        prefix: Current name prefix for exclusion pattern matching

    Returns:
        Module with BitLinear replaced by FP8BitLinear where appropriate
    """
    from wrinklefree.quantization.fp8_gemm import FP8Config

    fp8_config = fp8_config or FP8Config()
    converted_count = 0
    skipped_count = 0

    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name

        if isinstance(child, BitLinear) and not isinstance(child, FP8BitLinear):
            # Create FP8BitLinear with same configuration
            new_linear = FP8BitLinear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias is not None,
                eps=child.eps,
                fp8_config=fp8_config,
                layer_name=full_name,
            )
            # Copy weights
            new_linear.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                new_linear.bias.data.copy_(child.bias.data)

            setattr(module, name, new_linear)

            if new_linear._use_fp8:
                converted_count += 1
            else:
                skipped_count += 1
        else:
            # Recurse into children
            sub_converted, sub_skipped = _convert_recursive(child, fp8_config, full_name)
            converted_count += sub_converted
            skipped_count += sub_skipped

    logger.info(f"FP8 conversion: {converted_count} layers converted, {skipped_count} skipped")
    return module


def _convert_recursive(
    module: nn.Module,
    fp8_config: "FP8Config",
    prefix: str,
) -> tuple[int, int]:
    """Recursive helper for convert_bitlinear_to_fp8.

    Returns:
        Tuple of (converted_count, skipped_count)
    """
    converted_count = 0
    skipped_count = 0

    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name

        if isinstance(child, BitLinear) and not isinstance(child, FP8BitLinear):
            new_linear = FP8BitLinear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias is not None,
                eps=child.eps,
                fp8_config=fp8_config,
                layer_name=full_name,
            )
            new_linear.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                new_linear.bias.data.copy_(child.bias.data)

            setattr(module, name, new_linear)

            if new_linear._use_fp8:
                converted_count += 1
            else:
                skipped_count += 1
        else:
            sub_converted, sub_skipped = _convert_recursive(child, fp8_config, full_name)
            converted_count += sub_converted
            skipped_count += sub_skipped

    return converted_count, skipped_count
