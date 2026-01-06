"""BitLinearLRC: BitLinear with Low-Rank Correction for post-training quantization recovery.

Based on paper: "Low-Rank Correction for Quantized LLMs" (arxiv 2412.07902)

The key idea is to add trainable low-rank matrices (U, V) that operate on
unquantized activations to correct quantization errors:

    output = W_quant @ Q_a(X) + U @ V^T @ X

Where:
    - W_quant: Ternary quantized weights (frozen)
    - Q_a(X): 8-bit quantized activations
    - U, V: Full-precision low-rank correction matrices (trainable)
    - X: Original unquantized activations

Training loss (computed in training package):
    L = ||W @ X - W_quant @ Q_a(X) - U @ V^T @ X||^2

QLRC Extension:
    Optionally quantize the U and V matrices themselves using bitsandbytes
    for memory efficiency during training (QLoRA-style).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from wf_arch.layers.bitlinear import BitLinear

if TYPE_CHECKING:
    import bitsandbytes as bnb

logger = logging.getLogger(__name__)


@dataclass
class QLRCConfig:
    """Configuration for quantized LRC adapters (QLoRA-style).

    When enabled, the LRC matrices (U, V) are stored in quantized format
    using bitsandbytes, reducing memory usage during training.

    Attributes:
        enabled: Whether to use quantized adapters.
        quantization_type: Quantization format - "Q4_0" (4-bit NF4) or "Q8_0" (8-bit).
        compute_dtype: Dtype for computation (e.g., "bfloat16", "float16").
        use_double_quant: Use double quantization for Q4 (reduces memory further).
    """

    enabled: bool = False
    quantization_type: Literal["Q4_0", "Q8_0"] = "Q4_0"
    compute_dtype: str = "bfloat16"
    use_double_quant: bool = True

    def get_bnb_config(self) -> dict:
        """Get bitsandbytes configuration kwargs."""
        compute_dtype = getattr(torch, self.compute_dtype)

        if self.quantization_type == "Q4_0":
            return {
                "layer_class": "Linear4bit",
                "quant_type": "nf4",
                "compute_dtype": compute_dtype,
                "compress_statistics": self.use_double_quant,
            }
        else:  # Q8_0
            return {
                "layer_class": "Linear8bitLt",
                "has_fp16_weights": False,
                "threshold": 6.0,
            }


class BitLinearLRC(BitLinear):
    """
    BitLinear with Low-Rank Correction.

    Extends BitLinear with trainable low-rank matrices U, V that correct
    quantization errors using unquantized activations.

    By default, only U and V are trainable (weights frozen). Set trainable_weight=True
    to enable gradient flow through quantized weights via STE for joint training.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: Explicit low-rank dimension k
        rank_percentage: Rank as fraction of min(in_features, out_features).
            Only used if rank is None. Default: 0.1 (10%)
        bias: Whether to use bias (typically False for BitNet)
        eps: Small constant for numerical stability
        keep_original_weight: If False, delete original weight after quantization
            to save memory. Default True for backward compatibility.
        trainable_weight: If True, enable gradient flow through quantized weights
            via STE for joint training. Uses more memory. Default False.
        qlrc_config: Optional QLRC config for quantized adapters (QLoRA-style).
            When enabled, U and V are stored as quantized linear layers.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: Optional[int] = None,
        rank_percentage: float = 0.1,
        bias: bool = False,
        eps: float = 1e-5,
        keep_original_weight: bool = True,
        trainable_weight: bool = False,
        qlrc_config: Optional[QLRCConfig] = None,
    ):
        super().__init__(in_features, out_features, bias=bias, eps=eps)

        # Validate flag combinations
        if trainable_weight and not keep_original_weight:
            raise ValueError(
                "trainable_weight=True requires keep_original_weight=True "
                "(STE needs original weights for gradient flow)"
            )

        # Compute rank from percentage if not explicitly provided
        if rank is None:
            rank = max(1, int(min(in_features, out_features) * rank_percentage))

        self.rank = rank
        self.rank_percentage = rank_percentage
        self.keep_original_weight = keep_original_weight
        self.trainable_weight = trainable_weight
        self.qlrc_config = qlrc_config
        self._use_quantized_lrc = qlrc_config is not None and qlrc_config.enabled

        # Freeze weights unless trainable_weight is True (for STE gradient flow)
        self.weight.requires_grad = trainable_weight
        if self.bias is not None:
            self.bias.requires_grad = False

        # Initialize LRC matrices - either quantized (bitsandbytes) or full precision
        if self._use_quantized_lrc:
            self._init_quantized_lrc()
        else:
            # Trainable low-rank correction matrices (full precision)
            # U @ V^T has shape (out_features, in_features)
            # U: (out_features, rank), V: (in_features, rank)
            self.lrc_U = nn.Parameter(torch.zeros(out_features, rank))
            self.lrc_V = nn.Parameter(torch.zeros(in_features, rank))

        # Pre-computed quantized weights (frozen, saves recomputation each forward)
        self.weight_quantized = nn.Parameter(
            torch.empty_like(self.weight), requires_grad=False
        )
        self.compute_quantized_weights()

    def _init_quantized_lrc(self) -> None:
        """Initialize quantized LRC layers using bitsandbytes."""
        try:
            import bitsandbytes as bnb
        except ImportError as e:
            raise ImportError(
                "Quantized LRC (QLRC) requires bitsandbytes. "
                "Install with: pip install bitsandbytes>=0.43.0"
            ) from e

        cfg = self.qlrc_config.get_bnb_config()
        layer_class = cfg.pop("layer_class")

        if layer_class == "Linear4bit":
            # V projection: x @ V -> (batch, seq, rank)
            # Linear4bit: weight shape (out_features, in_features)
            self.lrc_V_linear = bnb.nn.Linear4bit(
                self.in_features,
                self.rank,
                bias=False,
                compute_dtype=cfg["compute_dtype"],
                quant_type=cfg["quant_type"],
                compress_statistics=cfg["compress_statistics"],
            )
            # U projection: result @ U^T -> (batch, seq, out_features)
            self.lrc_U_linear = bnb.nn.Linear4bit(
                self.rank,
                self.out_features,
                bias=False,
                compute_dtype=cfg["compute_dtype"],
                quant_type=cfg["quant_type"],
                compress_statistics=cfg["compress_statistics"],
            )
        else:  # Linear8bitLt
            self.lrc_V_linear = bnb.nn.Linear8bitLt(
                self.in_features,
                self.rank,
                bias=False,
                has_fp16_weights=cfg["has_fp16_weights"],
                threshold=cfg["threshold"],
            )
            self.lrc_U_linear = bnb.nn.Linear8bitLt(
                self.rank,
                self.out_features,
                bias=False,
                has_fp16_weights=cfg["has_fp16_weights"],
                threshold=cfg["threshold"],
            )

        # Initialize with zeros (default behavior)
        # Note: bitsandbytes layers initialize randomly, so we need to zero them
        with torch.no_grad():
            self.lrc_V_linear.weight.zero_()
            self.lrc_U_linear.weight.zero_()

    def compute_quantized_weights(self) -> None:
        """Pre-compute and cache the quantized weights.

        Call this after modifying self.weight. If keep_original_weight=False,
        this will also delete the original weight to save memory.

        Note: init_lrc_from_svd() requires the original weight, so call that
        before setting keep_original_weight=False.
        """
        if self.weight.numel() == 0:
            raise RuntimeError(
                "Cannot compute quantized weights: original weight was deleted. "
                "This layer was created with keep_original_weight=False."
            )

        with torch.no_grad():
            self.weight_quantized.data.copy_(self.weight_quant(self.weight))

        if not self.keep_original_weight:
            # Replace weight with empty tensor to save memory
            # Keep as parameter so state_dict structure is unchanged
            self.weight = nn.Parameter(
                torch.empty(0, device=self.weight.device, dtype=self.weight.dtype),
                requires_grad=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with low-rank correction.

        output = W_quant @ Q_a(X) + U @ V^T @ X

        When trainable_weight=False (default):
            Uses pre-computed quantized weights (memory efficient, no weight gradients)

        When trainable_weight=True:
            Computes quantization on-the-fly with STE for gradient flow to weights

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        if self.trainable_weight:
            # Compute quantization on-the-fly with STE for gradient flow
            w = self.weight.to(x.dtype)
            w_quant = w + (self.weight_quant(w) - w).detach()  # STE
            x_quant = x + (self.activation_quant(x) - x).detach()  # STE
        else:
            # Use pre-computed quantized weights (memory efficient)
            w_quant = self.weight_quantized.to(x.dtype)
            x_quant = self.activation_quant(x)

        quant_output = F.linear(x_quant, w_quant, self.bias)

        # Low-rank correction on UNQUANTIZED activations
        if self._use_quantized_lrc:
            # Quantized LRC path using bitsandbytes layers
            vt_x = self.lrc_V_linear(x)  # (batch, seq, rank)
            lrc_output = self.lrc_U_linear(vt_x)  # (batch, seq, out_features)
        else:
            # Full-precision LRC path
            # Efficient computation: U @ (V^T @ X) instead of (U @ V^T) @ X
            # V^T @ X: F.linear with V.t() as weight
            #   weight shape for F.linear: (out_features, in_features) = (rank, in_features)
            #   input: (batch, seq, in_features) -> output: (batch, seq, rank)
            # U @ result: F.linear with U as weight
            #   weight shape: (out_features, rank)
            #   input: (batch, seq, rank) -> output: (batch, seq, out_features)
            lrc_U = self.lrc_U.to(x.dtype)
            lrc_V = self.lrc_V.to(x.dtype)
            vt_x = F.linear(x, lrc_V.t())  # V.t(): (rank, in) -> output: (batch, seq, rank)
            lrc_output = F.linear(vt_x, lrc_U)  # U: (out, rank) -> output: (batch, seq, out)

        return quant_output + lrc_output

    def init_lrc_from_svd(self, original_weight: torch.Tensor) -> None:
        """
        Initialize U, V using SVD of the quantization residual (W - W_quant).

        This provides a better starting point than zeros by capturing the
        largest singular components of the quantization error.

        Args:
            original_weight: Original unquantized weight tensor
        """
        with torch.no_grad():
            w_quant = self.weight_quant(original_weight.to(self.weight.dtype))
            residual = original_weight.to(self.weight.dtype) - w_quant

            # SVD: residual = U @ S @ V^T
            U, S, Vh = torch.linalg.svd(residual.float(), full_matrices=False)

            # Take top-k components, distribute sqrt(S) to both U and V
            k = min(self.rank, len(S))
            sqrt_S = S[:k].sqrt()

            # U_init: (out_features, rank), V_init: (in_features, rank)
            U_init = (U[:, :k] * sqrt_S.unsqueeze(0)).to(original_weight.dtype)
            V_init = (Vh[:k, :].t() * sqrt_S.unsqueeze(0)).to(original_weight.dtype)

            if self._use_quantized_lrc:
                # For quantized LRC, set weights on the linear layers
                # lrc_V_linear: (in_features, rank) -> weight is (rank, in_features)
                # lrc_U_linear: (rank, out_features) -> weight is (out_features, rank)
                self.lrc_V_linear.weight.data[:, :k].copy_(V_init.t())
                self.lrc_U_linear.weight.data[:, :k].copy_(U_init)
            else:
                self.lrc_U.data[:, :k].copy_(U_init)
                self.lrc_V.data[:, :k].copy_(V_init)

    def get_lrc_weights_dequantized(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get dequantized U and V weights for GGUF export or inspection.

        For QLRC, this dequantizes the bitsandbytes weights back to full precision.
        For full-precision LRC, this returns the weights as-is.

        Returns:
            Tuple of (U, V) tensors:
                - U: shape (out_features, rank)
                - V: shape (in_features, rank)
        """
        if self._use_quantized_lrc:
            import bitsandbytes as bnb

            # Get weights from bitsandbytes layers
            # lrc_V_linear: weight is (rank, in_features)
            # lrc_U_linear: weight is (out_features, rank)
            if hasattr(self.lrc_V_linear.weight, "quant_state"):
                # 4-bit quantized - need to dequantize
                V_weight = bnb.functional.dequantize_4bit(
                    self.lrc_V_linear.weight.data,
                    self.lrc_V_linear.weight.quant_state,
                )
                U_weight = bnb.functional.dequantize_4bit(
                    self.lrc_U_linear.weight.data,
                    self.lrc_U_linear.weight.quant_state,
                )
            else:
                # 8-bit or not yet quantized - just get the weights
                V_weight = self.lrc_V_linear.weight.data.float()
                U_weight = self.lrc_U_linear.weight.data.float()

            # V_weight is (rank, in_features), need to transpose to (in_features, rank)
            return U_weight, V_weight.t()
        else:
            return self.lrc_U.data.clone(), self.lrc_V.data.clone()

    def extra_repr(self) -> str:
        base = (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, bias={self.bias is not None}"
        )
        if self._use_quantized_lrc:
            base += f", qlrc={self.qlrc_config.quantization_type}"
        return base


def convert_bitlinear_to_lrc(
    module: nn.Module,
    rank: Optional[int] = None,
    rank_percentage: float = 0.1,
    init_method: str = "zeros",
    exclude_names: Optional[list[str]] = None,
    keep_original_weight: bool = True,
    trainable_weight: bool = False,
    qlrc_config: Optional[QLRCConfig] = None,
) -> nn.Module:
    """
    Convert all BitLinear layers in a module to BitLinearLRC.

    This recursively replaces BitLinear layers with BitLinearLRC, preserving
    weights and freezing them. Only the new LRC matrices (U, V) are trainable
    by default.

    Args:
        module: Module containing BitLinear layers
        rank: Explicit rank for all layers (overrides rank_percentage)
        rank_percentage: Rank as fraction of min(in, out) per layer
        init_method: "zeros" (default) or "svd_residual"
        exclude_names: Layer names to exclude from conversion
        keep_original_weight: If False, delete original weights after
            quantization to save memory. Default True for compatibility.
        trainable_weight: If True, enable gradient flow through quantized
            weights via STE for joint training. Default False.
        qlrc_config: Optional QLRC config for quantized adapters.
            When enabled, U and V are stored as quantized linear layers.

    Returns:
        Module with BitLinear layers replaced by BitLinearLRC
    """
    exclude_names = exclude_names or []

    for name, child in module.named_children():
        if name in exclude_names:
            continue

        if isinstance(child, BitLinear) and not isinstance(child, BitLinearLRC):
            original_weight = child.weight.data.clone()

            # Always create with keep_original_weight=True initially so we can
            # copy weights. We'll update and recompute after.
            lrc_layer = BitLinearLRC(
                in_features=child.in_features,
                out_features=child.out_features,
                rank=rank,
                rank_percentage=rank_percentage,
                bias=child.bias is not None,
                eps=child.eps,
                qlrc_config=qlrc_config,
                keep_original_weight=True,  # Temporary, may change below
                trainable_weight=trainable_weight,
            )

            # Copy original weights (they will be frozen)
            # Preserve dtype/device for FSDP compatibility
            lrc_layer.weight.data = child.weight.data.clone()
            if child.bias is not None:
                lrc_layer.bias.data = child.bias.data.clone()

            # Initialize LRC matrices (must be before compute_quantized_weights
            # if keep_original_weight=False, since SVD needs original weight)
            if init_method == "svd_residual":
                lrc_layer.init_lrc_from_svd(original_weight)
            # else: zeros (already initialized in __init__)

            # Now set the actual keep_original_weight and recompute
            lrc_layer.keep_original_weight = keep_original_weight
            lrc_layer.compute_quantized_weights()

            setattr(module, name, lrc_layer)
        else:
            convert_bitlinear_to_lrc(
                child,
                rank=rank,
                rank_percentage=rank_percentage,
                init_method=init_method,
                exclude_names=exclude_names,
                keep_original_weight=keep_original_weight,
                trainable_weight=trainable_weight,
                qlrc_config=qlrc_config,
            )

    return module


def freeze_model_except_lrc(model: nn.Module) -> dict[str, int]:
    """
    Ensure ONLY LRC parameters (lrc_U, lrc_V, or lrc_U_linear, lrc_V_linear) are trainable.

    This is a safety function to call after model setup to guarantee
    that only the low-rank correction matrices are trained.

    Args:
        model: Model to freeze

    Returns:
        Dict with counts: {"trainable": N, "frozen": M}
    """
    trainable_count = 0
    frozen_count = 0

    # LRC parameter patterns:
    # - Full precision: lrc_U, lrc_V (nn.Parameter)
    # - Quantized (QLRC): lrc_U_linear.weight, lrc_V_linear.weight (bnb.nn.Linear)
    lrc_patterns = ("lrc_U", "lrc_V", "lrc_U_linear", "lrc_V_linear")

    for name, param in model.named_parameters():
        if any(pattern in name for pattern in lrc_patterns):
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False
            frozen_count += param.numel()

    logger.info(
        f"LRC freeze complete: {trainable_count:,} trainable params, "
        f"{frozen_count:,} frozen params"
    )

    return {"trainable": trainable_count, "frozen": frozen_count}


def get_lrc_stats(model: nn.Module) -> dict[str, int]:
    """
    Get statistics about LRC layers in a model.

    Args:
        model: Model to analyze

    Returns:
        Dict with stats about LRC layers
    """
    stats = {
        "num_lrc_layers": 0,
        "num_qlrc_layers": 0,
        "total_lrc_params": 0,
        "total_frozen_params": 0,
        "average_rank": 0,
        "ranks": [],
        "qlrc_quantization_type": None,
    }

    for module in model.modules():
        if isinstance(module, BitLinearLRC):
            stats["num_lrc_layers"] += 1

            if module._use_quantized_lrc:
                stats["num_qlrc_layers"] += 1
                stats["qlrc_quantization_type"] = module.qlrc_config.quantization_type
                # For QLRC, count params from the linear layers
                stats["total_lrc_params"] += (
                    module.lrc_U_linear.weight.numel()
                    + module.lrc_V_linear.weight.numel()
                )
            else:
                stats["total_lrc_params"] += module.lrc_U.numel() + module.lrc_V.numel()

            # Use weight_quantized for frozen param count (weight may be empty)
            stats["total_frozen_params"] += module.weight_quantized.numel()
            if module.bias is not None:
                stats["total_frozen_params"] += module.bias.numel()
            stats["ranks"].append(module.rank)

    if stats["ranks"]:
        stats["average_rank"] = sum(stats["ranks"]) / len(stats["ranks"])

    return stats
