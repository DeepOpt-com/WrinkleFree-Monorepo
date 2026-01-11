"""BitLinearSalient: BitLinear with FP16 salient columns for activation-aware quantization.

Based on AWQ (Activation-aware Weight Quantization): https://arxiv.org/abs/2306.00978

The key insight from AWQ is that a small fraction (0.1-1%) of weight columns contribute
disproportionately to output quality. These "salient" columns are identified by:

    saliency[col] = mean(|activation[:, col]|) * ||weight[:, col]||_2

This module keeps salient columns in FP16 while quantizing the rest to ternary:

    output = W_salient @ X_salient + W_quant @ Q_a(X_nonsalient)

Where:
    - W_salient: FP16 weights for salient columns (trainable)
    - W_quant: Ternary quantized weights for non-salient columns (trainable via STE)
    - X_salient: Unquantized activations for salient columns
    - Q_a(X_nonsalient): 8-bit quantized activations for non-salient columns

This is simpler than LRC (no U/V matrices) but requires calibration to identify salient columns.

Related work:
    - SqueezeLLM: https://arxiv.org/abs/2306.07629 (dense-and-sparse quantization)
    - SpQR: https://arxiv.org/abs/2306.03078 (sparse-quantized representation)
    - OWQ: https://arxiv.org/abs/2306.02272 (outlier-aware weight quantization)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from wf_arch.layers.bitlinear import BitLinear
from wf_arch.quantization.lambda_warmup import get_current_lambda

logger = logging.getLogger(__name__)


@dataclass
class SalientConfig:
    """Configuration for salient column selection.

    Attributes:
        ratio: Fraction of columns to keep in FP16 (default 0.01 = 1%)
        calibration_samples: Number of calibration samples for saliency detection
    """

    ratio: float = 0.01  # 1% salient columns
    calibration_samples: int = 128


class BitLinearSalient(BitLinear):
    """
    BitLinear with FP16 salient columns.

    Extends BitLinear to keep ~1% of columns in full precision based on
    AWQ-style activation-aware saliency scoring:

        saliency[col] = mean(|activation[:, col]|) * ||weight[:, col]||_2

    Forward pass (after calibration):
        - Salient columns: FP16 weight @ FP16 activation (no quantization)
        - Non-salient columns: Ternary weight @ 8-bit quantized activation
        - Final output: sum of both paths

    Before calibration, behaves exactly like standard BitLinear.

    Args:
        in_features: Size of input features
        out_features: Size of output features
        bias: Whether to include bias
        eps: Small constant for numerical stability
        salient_ratio: Fraction of columns to keep in FP16 (default 0.01)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        eps: float = 1e-5,
        salient_ratio: float = 0.01,
    ):
        super().__init__(in_features, out_features, bias=bias, eps=eps)
        self.salient_ratio = salient_ratio

        # Number of salient columns (at least 1)
        self.num_salient = max(1, int(in_features * salient_ratio))

        # Register buffer for salient column indices (set during calibration)
        # Shape: (num_salient,) - indices of columns to keep in FP16
        self.register_buffer(
            "salient_indices",
            torch.zeros(self.num_salient, dtype=torch.long),
        )

        # Register buffer for non-salient indices (complement set)
        # Shape: (in_features - num_salient,)
        self.register_buffer(
            "nonsalient_indices",
            torch.arange(in_features, dtype=torch.long),
        )

        # Track whether calibration has been performed
        self.register_buffer(
            "is_calibrated",
            torch.tensor(False, dtype=torch.bool),
        )

        # Store saliency scores for debugging/analysis
        self.register_buffer(
            "saliency_scores",
            torch.zeros(in_features),
        )

    def set_salient_columns(
        self,
        indices: torch.Tensor,
        saliency_scores: Optional[torch.Tensor] = None,
    ) -> None:
        """Set the salient column indices after calibration.

        Args:
            indices: Tensor of column indices to keep in FP16. Shape: (num_salient,)
            saliency_scores: Optional full saliency scores for all columns (for debugging)
        """
        if indices.numel() != self.num_salient:
            raise ValueError(
                f"Expected {self.num_salient} salient columns, got {indices.numel()}"
            )

        device = self.salient_indices.device
        self.salient_indices.copy_(indices.to(device))

        # Compute non-salient indices (complement set)
        all_indices = torch.arange(self.in_features, device=device)
        mask = torch.ones(self.in_features, dtype=torch.bool, device=device)
        mask[self.salient_indices] = False
        nonsalient = all_indices[mask]
        # Re-register buffer with correct size (may differ from init size)
        self.register_buffer("nonsalient_indices", nonsalient)

        # Store saliency scores if provided
        if saliency_scores is not None:
            self.saliency_scores.copy_(saliency_scores.to(device))

        self.is_calibrated.fill_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with mixed-precision computation.

        Before calibration: Standard BitLinear behavior (full ternary quantization)

        After calibration:
            - Salient columns: FP16 weight @ FP16 activation (no quantization)
            - Non-salient columns: Ternary weight @ 8-bit quantized activation
            - Uses STE for gradient flow through quantization

        Supports lambda warmup for gradual quantization transition during training.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        lambda_val = get_current_lambda()

        # Cast weight to input dtype for mixed precision compatibility
        w = self.weight.to(x.dtype)

        if not self.is_calibrated:
            # Before calibration: use standard BitLinear quantization
            w_quant = w + lambda_val * (self.weight_quant(w) - w).detach()
            x_quant = x + lambda_val * (self.activation_quant(x) - x).detach()
            bias = self.bias.to(x.dtype) if self.bias is not None else None
            return F.linear(x_quant, w_quant, bias)

        # After calibration: mixed precision forward pass
        # Note: Debug logging removed for torch.compile compatibility

        # Split weight matrix by columns
        w_salient = w[:, self.salient_indices]  # (out_features, num_salient)
        w_nonsalient = w[:, self.nonsalient_indices]  # (out_features, in - num_salient)

        # Split input by columns
        x_salient = x[..., self.salient_indices]  # (..., num_salient)
        x_nonsalient = x[..., self.nonsalient_indices]  # (..., in - num_salient)

        # === Salient path: FP16 (no quantization) ===
        # These columns are kept at full precision for maximum quality
        # Lambda warmup still applies: at lambda=0, everything is FP16 anyway
        output_salient = F.linear(x_salient, w_salient)  # (..., out_features)

        # === Non-salient path: Ternary quantization with STE ===
        # Use standard BitLinear quantization for the rest
        # Guard against empty non-salient indices (all columns salient)
        if self.nonsalient_indices.numel() > 0:
            w_nonsalient_quant = w_nonsalient + lambda_val * (
                self._weight_quant_subset(w_nonsalient) - w_nonsalient
            ).detach()
            x_nonsalient_quant = x_nonsalient + lambda_val * (
                self._activation_quant_subset(x_nonsalient) - x_nonsalient
            ).detach()
            output_nonsalient = F.linear(
                x_nonsalient_quant, w_nonsalient_quant
            )  # (..., out_features)
            output = output_salient + output_nonsalient
        else:
            # All columns are salient - no quantized path needed
            output = output_salient

        if self.bias is not None:
            output = output + self.bias.to(x.dtype)

        return output

    def _weight_quant_subset(self, w: torch.Tensor) -> torch.Tensor:
        """Quantize a subset of weights to ternary values.

        Uses the scale from the subset, not the full weight matrix.
        This ensures proper quantization for the non-salient columns only.

        Args:
            w: Weight tensor subset

        Returns:
            Quantized weight tensor
        """
        scale = 1.0 / w.abs().mean().clamp(min=self.eps)
        return (w * scale).round().clamp(-1, 1) / scale

    def _activation_quant_subset(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize a subset of activations to 8-bit.

        Uses per-token absmax quantization on the subset.

        Args:
            x: Activation tensor subset of shape (..., features)

        Returns:
            Quantized activation tensor
        """
        scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=self.eps)
        return (x * scale).round().clamp(-128, 127) / scale

    def get_salient_stats(self) -> dict:
        """Get statistics about salient columns in this layer."""
        stats = {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "num_salient": self.num_salient,
            "salient_ratio": self.salient_ratio,
            "is_calibrated": self.is_calibrated.item(),
        }

        if self.is_calibrated:
            stats["salient_indices"] = self.salient_indices.tolist()
            stats["top_saliency_scores"] = self.saliency_scores[
                self.salient_indices
            ].tolist()

        return stats

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Override to handle buffer size mismatch during checkpoint loading.

        The nonsalient_indices buffer changes size after calibration (from in_features
        to in_features - num_salient), so we need to resize it before loading.
        """
        nonsalient_key = prefix + "nonsalient_indices"
        if nonsalient_key in state_dict:
            # Resize buffer to match checkpoint
            loaded_size = state_dict[nonsalient_key].numel()
            if loaded_size != self.nonsalient_indices.numel():
                self.nonsalient_indices = torch.zeros(
                    loaded_size,
                    dtype=self.nonsalient_indices.dtype,
                    device=self.nonsalient_indices.device,
                )

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"salient_ratio={self.salient_ratio}, num_salient={self.num_salient}, "
            f"calibrated={self.is_calibrated.item()}"
        )


def convert_bitlinear_to_salient(
    module: nn.Module,
    salient_ratio: float = 0.01,
    salient_indices: Optional[Dict[str, torch.Tensor]] = None,
    saliency_scores: Optional[Dict[str, torch.Tensor]] = None,
    exclude_names: Optional[list[str]] = None,
) -> nn.Module:
    """
    Convert all BitLinear layers in a module to BitLinearSalient.

    This recursively replaces BitLinear layers with BitLinearSalient,
    preserving weights. If salient_indices are provided (from calibration),
    they are applied to each layer.

    Args:
        module: Module containing BitLinear layers
        salient_ratio: Fraction of columns to keep in FP16
        salient_indices: Pre-computed salient indices from calibration.
            Dict mapping layer names to index tensors.
            If None, layers will need calibration later.
        saliency_scores: Optional saliency scores from calibration.
            Dict mapping layer names to score tensors.
        exclude_names: Layer names to exclude from conversion

    Returns:
        Module with BitLinear layers replaced by BitLinearSalient
    """
    exclude_names = exclude_names or []
    layer_count = 0

    # Build full name mapping for recursive conversion
    def _convert_recursive(
        mod: nn.Module,
        prefix: str = "",
    ) -> None:
        nonlocal layer_count

        for name, child in mod.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            if name in exclude_names or full_name in exclude_names:
                continue

            if isinstance(child, BitLinear) and not isinstance(
                child, BitLinearSalient
            ):
                layer_count += 1
                logger.info(
                    f"[{layer_count}] Converting {full_name} "
                    f"({child.out_features}x{child.in_features}) to BitLinearSalient"
                )

                # Create BitLinearSalient with same configuration
                salient_layer = BitLinearSalient(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=child.bias is not None,
                    eps=child.eps,
                    salient_ratio=salient_ratio,
                )

                # Move layer to same device as original BEFORE copying weights
                # This ensures buffers (salient_indices, etc.) are on correct device
                device = child.weight.device
                salient_layer = salient_layer.to(device)

                # Copy weights (preserve dtype/device for FSDP compatibility)
                salient_layer.weight.data = child.weight.data.clone()
                if child.bias is not None:
                    salient_layer.bias.data = child.bias.data.clone()

                # Apply pre-computed salient indices if available
                if salient_indices is not None and full_name in salient_indices:
                    scores = (
                        saliency_scores.get(full_name) if saliency_scores else None
                    )
                    salient_layer.set_salient_columns(
                        salient_indices[full_name],
                        saliency_scores=scores,
                    )
                    logger.info(
                        f"    Applied {salient_layer.num_salient} salient columns"
                    )

                setattr(mod, name, salient_layer)
            else:
                # Recursively process child modules
                _convert_recursive(child, full_name)

    _convert_recursive(module)

    logger.info(f"Converted {layer_count} layers to BitLinearSalient")
    return module


def get_salient_stats(model: nn.Module) -> dict:
    """
    Get statistics about BitLinearSalient layers in a model.

    Args:
        model: Model to analyze

    Returns:
        Dict with statistics about salient layers
    """
    stats = {
        "num_salient_layers": 0,
        "num_calibrated_layers": 0,
        "total_salient_columns": 0,
        "total_columns": 0,
        "average_salient_ratio": 0.0,
        "layers": [],
    }

    for name, module in model.named_modules():
        if isinstance(module, BitLinearSalient):
            stats["num_salient_layers"] += 1
            stats["total_salient_columns"] += module.num_salient
            stats["total_columns"] += module.in_features

            if module.is_calibrated:
                stats["num_calibrated_layers"] += 1

            stats["layers"].append(
                {
                    "name": name,
                    "in_features": module.in_features,
                    "out_features": module.out_features,
                    "num_salient": module.num_salient,
                    "is_calibrated": module.is_calibrated.item(),
                }
            )

    if stats["total_columns"] > 0:
        stats["average_salient_ratio"] = (
            stats["total_salient_columns"] / stats["total_columns"]
        )

    return stats
