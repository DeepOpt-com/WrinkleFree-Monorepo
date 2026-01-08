"""LoRAAdapter: Composable Low-Rank Adaptation wrapper for any linear layer.

This module provides a clean wrapper pattern for adding LoRA (Low-Rank Adaptation)
to any linear layer, including BitLinear and BitLinearSalient. This enables
orthogonal composition of features:

    model = convert_bitlinear_to_salient(model)  # Step 1: Add saliency
    model = add_lora_to_model(model, config)     # Step 2: Add LoRA (works with any base)

The key design principle is that LoRA operates on the UNQUANTIZED activations
to correct quantization errors:

    output = base_layer(x) + lora_B(lora_A(x)) * scaling

Where:
    - base_layer: Any linear layer (BitLinear, BitLinearSalient, nn.Linear, etc.)
    - lora_A: Down-projection (in_features -> rank)
    - lora_B: Up-projection (rank -> out_features)
    - scaling: alpha / rank

Based on:
- LoRA: https://arxiv.org/abs/2106.09685
- QA-LoRA: https://arxiv.org/abs/2309.14717
- Low-Rank Correction: https://arxiv.org/abs/2412.07902
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapters.

    Attributes:
        rank: Explicit rank dimension. Mutually exclusive with rank_percentage.
        rank_percentage: Rank as fraction of min(in_features, out_features).
            Default 0.1 (10%). Ignored if rank is set.
        alpha: LoRA scaling factor. Output is scaled by alpha/rank.
        dropout: Dropout probability for LoRA path. Default 0.0.
        init_method: Initialization method for LoRA matrices.
            - "kaiming": LoRA-style (A=Kaiming, B=zeros). Fast, good gradients.
            - "svd_residual": SVD of quantization error. Best accuracy.
            - "zeros": Both A and B zeros. Not recommended (no gradients).
        quantized: Enable QA-LoRA style quantized adapters using STE.
        quant_bits: Quantization bits (4 or 8) when quantized=True.
        quant_group_size: Group size for group-wise quantization.
        target_modules: Regex patterns for layers to apply LoRA.
            If None, applies to all BitLinear/BitLinearSalient layers.
    """

    rank: Optional[int] = None
    rank_percentage: float = 0.1
    alpha: float = 1.0
    dropout: float = 0.0
    init_method: str = "kaiming"
    quantized: bool = False
    quant_bits: int = 4
    quant_group_size: int = 32
    target_modules: Optional[List[str]] = None

    def __post_init__(self):
        if self.rank is not None and self.rank_percentage != 0.1:
            raise ValueError(
                "Cannot specify both rank and rank_percentage. Use one or the other."
            )
        if self.init_method not in ("kaiming", "svd_residual", "zeros"):
            raise ValueError(
                f"init_method must be 'kaiming', 'svd_residual', or 'zeros', "
                f"got '{self.init_method}'"
            )


class QuantizedLinearSTE(nn.Module):
    """Linear layer with STE (Straight-Through Estimator) quantization.

    Enables training of quantized adapters by:
    1. Storing weights in full precision (for gradient updates)
    2. Quantizing on-the-fly during forward pass
    3. Using STE so gradients bypass quantization during backward pass

    Based on QA-LoRA: https://arxiv.org/abs/2309.14717 (ICLR 2024)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 4,
        group_size: int = 32,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_quant = self._quantize_ste(self.weight)
        return F.linear(x, w_quant, self.bias)

    def _quantize_ste(self, w: torch.Tensor) -> torch.Tensor:
        """Group-wise symmetric quantization with Straight-Through Estimator."""
        original_shape = w.shape
        numel = w.numel()

        if numel % self.group_size != 0:
            pad_size = self.group_size - (numel % self.group_size)
            w_padded = F.pad(w.view(-1), (0, pad_size))
        else:
            w_padded = w.view(-1)
            pad_size = 0

        w_grouped = w_padded.view(-1, self.group_size)
        qmax = 2 ** (self.bits - 1) - 1
        scale = w_grouped.abs().amax(dim=1, keepdim=True) / qmax
        scale = scale.clamp(min=1e-8)

        w_int = (w_grouped / scale).round().clamp(-qmax - 1, qmax)
        w_dequant = w_int * scale

        # STE: forward uses dequant, backward uses original
        w_ste = w_grouped + (w_dequant - w_grouped).detach()

        w_ste = w_ste.view(-1)
        if pad_size > 0:
            w_ste = w_ste[:-pad_size]

        return w_ste.view(original_shape)


class LoRAAdapter(nn.Module):
    """Composable LoRA adapter that wraps any linear layer.

    This wrapper adds low-rank adaptation to any base layer without modifying it.
    The LoRA correction is computed on unquantized activations and added to
    the base layer's output.

    Example:
        >>> base = BitLinearSalient(768, 768)
        >>> lora = LoRAAdapter(base, LoRAConfig(rank=32))
        >>> output = lora(x)  # base(x) + lora_correction(x)

    Attributes:
        base_layer: The wrapped linear layer (BitLinear, BitLinearSalient, etc.)
        rank: The low-rank dimension
        scaling: Output scaling factor (alpha / rank)
    """

    def __init__(
        self,
        base_layer: nn.Module,
        config: LoRAConfig,
        original_weight: Optional[torch.Tensor] = None,
    ):
        """Initialize LoRA adapter.

        Args:
            base_layer: Linear layer to wrap (must have in_features, out_features)
            config: LoRA configuration
            original_weight: Original unquantized weight for SVD initialization.
                Required if config.init_method == "svd_residual".
        """
        super().__init__()

        self.base_layer = base_layer
        self.config = config

        # Get dimensions from base layer
        if hasattr(base_layer, "in_features"):
            in_features = base_layer.in_features
            out_features = base_layer.out_features
        elif hasattr(base_layer, "weight"):
            out_features, in_features = base_layer.weight.shape
        else:
            raise ValueError(
                f"Cannot determine dimensions from base_layer {type(base_layer)}"
            )

        self.in_features = in_features
        self.out_features = out_features

        # Compute rank
        if config.rank is not None:
            rank = config.rank
        else:
            rank = max(1, int(min(in_features, out_features) * config.rank_percentage))
        self.rank = rank

        # Scaling factor
        self.scaling = config.alpha / rank

        # Dropout
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        # Initialize LoRA matrices
        if config.quantized:
            self.lora_A = QuantizedLinearSTE(
                in_features, rank,
                bits=config.quant_bits,
                group_size=config.quant_group_size,
                bias=False,
            )
            self.lora_B = QuantizedLinearSTE(
                rank, out_features,
                bits=config.quant_bits,
                group_size=config.quant_group_size,
                bias=False,
            )
        else:
            self.lora_A = nn.Linear(in_features, rank, bias=False)
            self.lora_B = nn.Linear(rank, out_features, bias=False)

        # Initialize weights based on method
        self._initialize_weights(config.init_method, original_weight)

    def _initialize_weights(
        self,
        method: str,
        original_weight: Optional[torch.Tensor] = None,
    ) -> None:
        """Initialize LoRA weights based on method."""
        with torch.no_grad():
            if method == "kaiming":
                # LoRA-style: A=Kaiming, B=zeros
                # Initial output is zero, but gradients flow immediately
                nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B.weight)

            elif method == "zeros":
                nn.init.zeros_(self.lora_A.weight)
                nn.init.zeros_(self.lora_B.weight)

            elif method == "svd_residual":
                if original_weight is None:
                    logger.warning(
                        "svd_residual init requires original_weight, "
                        "falling back to kaiming"
                    )
                    self._initialize_weights("kaiming", None)
                    return

                self._init_from_svd(original_weight)

    def _init_from_svd(
        self,
        original_weight: torch.Tensor,
        fast_svd: bool = True,
        fast_svd_oversampling: int = 10,
        fast_svd_niter: int = 2,
    ) -> None:
        """Initialize from SVD of quantization residual."""
        # Get the effective quantized weight from base layer
        effective_quant_weight = self._get_effective_quantized_weight()

        if effective_quant_weight is None:
            logger.warning(
                "Could not get effective quantized weight from base layer, "
                "using original weight directly for SVD"
            )
            residual = original_weight.float()
        else:
            residual = (original_weight - effective_quant_weight).float()

        k = self.rank

        if fast_svd:
            q = min(k + fast_svd_oversampling, min(residual.shape))
            U, S, V = torch.svd_lowrank(residual, q=q, niter=fast_svd_niter)
        else:
            U, S, Vh = torch.linalg.svd(residual, full_matrices=False)
            V = Vh.t()

        k = min(k, len(S))
        sqrt_S = S[:k].sqrt()

        # A: (rank, in_features), B: (out_features, rank)
        # U: (out_features, k), V: (in_features, k)
        A_init = (V[:, :k] * sqrt_S.unsqueeze(0)).t()  # (k, in_features)
        B_init = U[:, :k] * sqrt_S.unsqueeze(0)  # (out_features, k)

        with torch.no_grad():
            # lora_A.weight: (rank, in_features)
            self.lora_A.weight.data[:k, :].copy_(A_init.to(self.lora_A.weight.dtype))
            # lora_B.weight: (out_features, rank)
            self.lora_B.weight.data[:, :k].copy_(B_init.to(self.lora_B.weight.dtype))

    def _get_effective_quantized_weight(self) -> Optional[torch.Tensor]:
        """Get the effective quantized weight from base layer.

        For BitLinearSalient, this includes the salient column handling.
        Returns None if unable to determine.
        """
        base = self.base_layer

        # Try to get pre-computed quantized weight
        if hasattr(base, "weight_quantized"):
            return base.weight_quantized.data

        # Try to compute it
        if hasattr(base, "weight_quant") and hasattr(base, "weight"):
            return base.weight_quant(base.weight)

        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA correction.

        output = base_layer(x) + lora_B(lora_A(x)) * scaling
        """
        base_output = self.base_layer(x)
        lora_output = self.lora_B(self.dropout(self.lora_A(x))) * self.scaling
        return base_output + lora_output

    def get_lora_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get LoRA weights (A, B) for export or inspection.

        Returns:
            Tuple of (A, B) tensors:
                - A: shape (rank, in_features)
                - B: shape (out_features, rank)
        """
        if self.config.quantized:
            with torch.no_grad():
                A = self.lora_A._quantize_ste(self.lora_A.weight)
                B = self.lora_B._quantize_ste(self.lora_B.weight)
            return A, B
        else:
            return self.lora_A.weight.data.clone(), self.lora_B.weight.data.clone()

    def extra_repr(self) -> str:
        quant_str = ""
        if self.config.quantized:
            quant_str = f", quantized={self.config.quant_bits}bit"
        return (
            f"rank={self.rank}, scaling={self.scaling:.4f}, "
            f"in={self.in_features}, out={self.out_features}{quant_str}"
        )


# =============================================================================
# Model-level utilities
# =============================================================================


def add_lora_to_model(
    model: nn.Module,
    config: LoRAConfig,
    original_weights: Optional[Dict[str, torch.Tensor]] = None,
) -> nn.Module:
    """Add LoRA adapters to layers matching target_modules patterns.

    Args:
        model: Model to add LoRA to
        config: LoRA configuration
        original_weights: Dict mapping layer names to original unquantized weights.
            Required for svd_residual initialization. Keys should match layer names.

    Returns:
        Model with LoRA adapters added
    """
    from wf_arch.layers.bitlinear import BitLinear
    from wf_arch.layers.bitlinear_salient import BitLinearSalient

    original_weights = original_weights or {}
    target_patterns = config.target_modules or []

    def _should_add_lora(name: str, module: nn.Module) -> bool:
        """Check if LoRA should be added to this module."""
        # Must be a BitLinear variant
        if not isinstance(module, (BitLinear, BitLinearSalient)):
            return False

        # Already wrapped
        if isinstance(module, LoRAAdapter):
            return False

        # Check target patterns (if specified)
        if target_patterns:
            return any(re.search(pattern, name) for pattern in target_patterns)

        return True

    # Collect layers to wrap (can't modify during iteration)
    layers_to_wrap = []
    for name, module in model.named_modules():
        if _should_add_lora(name, module):
            layers_to_wrap.append((name, module))

    # Wrap layers
    for name, module in layers_to_wrap:
        original_weight = original_weights.get(name)

        # For SVD init, try to get original weight from module
        if config.init_method == "svd_residual" and original_weight is None:
            if hasattr(module, "weight") and module.weight.numel() > 0:
                original_weight = module.weight.data.clone()

        lora_wrapper = LoRAAdapter(module, config, original_weight)

        # Replace in parent module
        if "." in name:
            parent_path, attr_name = name.rsplit(".", 1)
            parent = model
            for part in parent_path.split("."):
                parent = getattr(parent, part)
        else:
            parent = model
            attr_name = name
        setattr(parent, attr_name, lora_wrapper)

        logger.info(f"Added LoRA to {name} (rank={lora_wrapper.rank})")

    logger.info(f"Added LoRA to {len(layers_to_wrap)} layers")
    return model


def freeze_base_model(model: nn.Module) -> Dict[str, int]:
    """Freeze all non-LoRA parameters.

    Only LoRA adapter parameters (lora_A, lora_B) will be trainable.
    Base layer weights, embeddings, and other parameters are frozen.

    Args:
        model: Model to freeze

    Returns:
        Dict with counts: {"trainable": N, "frozen": M}
    """
    trainable_count = 0
    frozen_count = 0

    lora_patterns = ("lora_A", "lora_B")

    for name, param in model.named_parameters():
        if any(pattern in name for pattern in lora_patterns):
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False
            frozen_count += param.numel()

    logger.info(
        f"Freeze complete: {trainable_count:,} trainable params, "
        f"{frozen_count:,} frozen params"
    )

    return {"trainable": trainable_count, "frozen": frozen_count}


def remove_lora_from_model(model: nn.Module) -> nn.Module:
    """Remove LoRA adapters from model, keeping only base layers.

    Args:
        model: Model with LoRA adapters

    Returns:
        Model with LoRA adapters removed (base layers exposed)
    """
    # Collect LoRA wrappers
    lora_wrappers = []
    for name, module in model.named_modules():
        if isinstance(module, LoRAAdapter):
            lora_wrappers.append((name, module))

    # Unwrap
    for name, wrapper in lora_wrappers:
        if "." in name:
            parent_path, attr_name = name.rsplit(".", 1)
            parent = model
            for part in parent_path.split("."):
                parent = getattr(parent, part)
        else:
            parent = model
            attr_name = name
        setattr(parent, attr_name, wrapper.base_layer)
        logger.info(f"Removed LoRA from {name}")

    return model


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """Merge LoRA weights into base layer weights.

    WARNING: This modifies the base layer weights in-place.
    Only use for FP16 inference or analysis. Cannot be used with
    ternary-quantized weights (merging FP16 into ternary is lossy).

    Args:
        model: Model with LoRA adapters

    Returns:
        Model with LoRA merged into base weights
    """
    for name, module in model.named_modules():
        if isinstance(module, LoRAAdapter):
            base = module.base_layer

            # Check if base is quantized
            if hasattr(base, "weight_quantized"):
                logger.warning(
                    f"Skipping merge for {name}: base layer uses quantized weights. "
                    "Merging FP16 LoRA into ternary weights would be lossy."
                )
                continue

            # Merge: W_new = W + B @ A * scaling
            A, B = module.get_lora_weights()
            delta = (B @ A) * module.scaling

            with torch.no_grad():
                base.weight.data.add_(delta.to(base.weight.dtype))

            logger.info(f"Merged LoRA weights for {name}")

    return remove_lora_from_model(model)


def get_lora_stats(model: nn.Module) -> Dict[str, Union[int, float, List]]:
    """Get statistics about LoRA layers in a model.

    Args:
        model: Model to analyze

    Returns:
        Dict with stats about LoRA layers
    """
    stats = {
        "num_lora_layers": 0,
        "num_quantized_lora": 0,
        "total_lora_params": 0,
        "total_base_params": 0,
        "average_rank": 0.0,
        "ranks": [],
        "layers": [],
    }

    for name, module in model.named_modules():
        if isinstance(module, LoRAAdapter):
            stats["num_lora_layers"] += 1

            if module.config.quantized:
                stats["num_quantized_lora"] += 1

            lora_params = (
                module.lora_A.weight.numel() + module.lora_B.weight.numel()
            )
            stats["total_lora_params"] += lora_params

            if hasattr(module.base_layer, "weight"):
                stats["total_base_params"] += module.base_layer.weight.numel()

            stats["ranks"].append(module.rank)
            stats["layers"].append({
                "name": name,
                "rank": module.rank,
                "in_features": module.in_features,
                "out_features": module.out_features,
                "quantized": module.config.quantized,
            })

    if stats["ranks"]:
        stats["average_rank"] = sum(stats["ranks"]) / len(stats["ranks"])

    return stats


def remap_legacy_checkpoint(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remap pre-LoRA checkpoint keys to wrapped structure.

    When a model is wrapped with LoRAAdapter, state dict keys change:
        model.layer.weight -> model.layer.base_layer.weight

    This function remaps old keys to new structure.

    Args:
        state_dict: State dict with old key structure

    Returns:
        State dict with keys remapped for LoRA-wrapped model
    """
    new_state_dict = {}

    for key, value in state_dict.items():
        # Check if this key should be remapped (heuristic: ends with .weight/.bias
        # and doesn't already have .base_layer in path)
        if ".base_layer." not in key:
            # Try to find corresponding LoRA-wrapped key
            # This is a simple heuristic - may need adjustment for specific models
            parts = key.rsplit(".", 1)
            if len(parts) == 2 and parts[1] in ("weight", "bias", "weight_quantized"):
                new_key = f"{parts[0]}.base_layer.{parts[1]}"
                new_state_dict[new_key] = value
                # Also keep original key for non-LoRA layers
                new_state_dict[key] = value
                continue

        new_state_dict[key] = value

    return new_state_dict
