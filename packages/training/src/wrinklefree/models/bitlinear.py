"""BitLinear: Quantized linear layer for BitNet 1.58-bit models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from wrinklefree.quantization.lambda_warmup import get_current_lambda
from wrinklefree.quantization.sparsity_warmup import get_current_sparsity
from wrinklefree.quantization.activation_sparse import detach_sparsify

if TYPE_CHECKING:
    from wrinklefree.quantization.saliency_curriculum import SaliencyCurriculum


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
        Supports Q-Sparse activation sparsification (arxiv.org/abs/2407.10969).

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        # Get lambda for gradual quantization (1.0 = full quant, 0.0 = no quant)
        lambda_val = get_current_lambda()
        # Get sparsity ratio (0.0 = dense, >0 = sparse)
        sparsity_ratio = get_current_sparsity()

        # Cast weight to input dtype for mixed precision compatibility
        w = self.weight.to(x.dtype)

        # Gradual quantization with lambda warmup
        # w_quant = w + lambda * (quant(w) - w) = (1-lambda)*w + lambda*quant(w)
        w_quant = w + lambda_val * (self.weight_quant(w) - w).detach()

        # Q-Sparse: Apply activation sparsity BEFORE quantization
        # Keeps top (1 - sparsity_ratio) activations by magnitude
        if sparsity_ratio > 0:
            x = detach_sparsify(x, sparsity_ratio, per_token=True)

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


class SaliencyAwareBitLinear(BitLinear):
    """
    BitLinear variant that supports saliency-aware mixed-precision quantization.

    When a SaliencyCurriculum is attached, salient columns (high L-inf norm) are
    kept in FP16 while non-salient columns are quantized to ternary.

    This allows gradual transition from mixed-precision to fully ternary during training.

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
        super().__init__(in_features, out_features, bias=bias, eps=eps)
        self._saliency_curriculum: Optional[SaliencyCurriculum] = None
        self._layer_name: Optional[str] = None

    def set_saliency_curriculum(
        self,
        curriculum: SaliencyCurriculum,
        layer_name: str,
    ) -> None:
        """
        Attach saliency curriculum for mixed-precision training.

        Args:
            curriculum: SaliencyCurriculum instance managing the training schedule
            layer_name: Unique identifier for this layer in the curriculum
        """
        self._saliency_curriculum = curriculum
        self._layer_name = layer_name
        curriculum.register_layer(layer_name, self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional saliency-aware mixed-precision and lambda warmup.

        During training with curriculum attached:
        - Updates saliency EMA tracking
        - Applies mixed-precision: salient columns in FP16, others in ternary
        - Uses STE for gradient flow through both branches

        During inference or when curriculum is disabled:
        - Falls back to standard BitLinear quantization

        Lambda warmup is applied in all cases for gradual quantization.
        Q-Sparse activation sparsification is applied before quantization.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        # Get lambda for gradual quantization (1.0 = full quant, 0.0 = no quant)
        lambda_val = get_current_lambda()
        # Get sparsity ratio (0.0 = dense, >0 = sparse)
        sparsity_ratio = get_current_sparsity()

        w = self.weight.to(x.dtype)

        if self._saliency_curriculum is not None and self.training:
            # Update saliency tracking (only every update_interval steps)
            self._saliency_curriculum.update_saliency(self._layer_name, w)

            # Get saliency mask: True = keep FP16, False = quantize
            saliency_mask = self._saliency_curriculum.get_saliency_mask(
                self._layer_name, w
            )

            # Mixed-precision quantization with STE and lambda warmup
            w_quant = self._saliency_aware_quant(w, saliency_mask, lambda_val)
        else:
            # Standard full quantization with lambda warmup
            w_quant = w + lambda_val * (self.weight_quant(w) - w).detach()

        # Q-Sparse: Apply activation sparsity BEFORE quantization
        if sparsity_ratio > 0:
            x = detach_sparsify(x, sparsity_ratio, per_token=True)

        # Quantize activations with STE and lambda warmup
        x_quant = x + lambda_val * (self.activation_quant(x) - x).detach()

        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x_quant, w_quant, bias)

    def _saliency_aware_quant(
        self,
        w: torch.Tensor,
        saliency_mask: torch.Tensor,
        lambda_val: float,
    ) -> torch.Tensor:
        """
        Apply mixed-precision quantization based on saliency mask with lambda warmup.

        Salient columns (mask=True) keep FP16 values.
        Non-salient columns (mask=False) get ternary quantized.

        Uses STE for gradient flow through both branches.
        Lambda warmup provides gradual transition to quantization.

        Args:
            w: Weight tensor of shape (out_features, in_features)
            saliency_mask: Boolean mask of shape (1, in_features)
                True = salient (keep FP16), False = quantize to ternary
            lambda_val: Lambda warmup value (0.0 = no quant, 1.0 = full quant)

        Returns:
            Mixed-precision quantized weights with STE gradient flow
        """
        # Expand mask to weight shape: (out_features, in_features)
        mask = saliency_mask.expand_as(w)

        # Quantize the non-salient columns
        w_quant_full = self.weight_quant(w)

        # Mix: salient columns use FP16, non-salient use ternary
        # torch.where: select from w where mask is True, else from w_quant_full
        w_mixed = torch.where(mask, w, w_quant_full)

        # Apply STE with lambda warmup: gradients flow to w for ALL columns
        return w + lambda_val * (w_mixed - w).detach()


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


def convert_bitlinear_to_saliency_aware(
    module: nn.Module,
    exclude_names: list[str] | None = None,
) -> nn.Module:
    """
    Convert all BitLinear layers in a module to SaliencyAwareBitLinear.

    This is used before Stage 1.9 when saliency curriculum is enabled.
    The conversion preserves all weights and layer configurations.

    Args:
        module: The module to convert
        exclude_names: List of layer names to exclude from conversion

    Returns:
        Module with BitLinear layers replaced by SaliencyAwareBitLinear
    """
    exclude_names = exclude_names or []

    for name, child in module.named_children():
        if name in exclude_names:
            continue

        if isinstance(child, BitLinear) and not isinstance(child, SaliencyAwareBitLinear):
            # Create SaliencyAwareBitLinear with same configuration
            new_linear = SaliencyAwareBitLinear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias is not None,
                eps=child.eps,
            )
            # Copy weights
            new_linear.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                new_linear.bias.data.copy_(child.bias.data)

            setattr(module, name, new_linear)
        else:
            convert_bitlinear_to_saliency_aware(child, exclude_names)

    return module
