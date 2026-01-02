"""Quantization utilities for BitNet 1.58-bit models."""

from wrinklefree.quantization.activation_quant import (
    activation_quantization_absmean,
    activation_quantization_per_tensor,
    activation_quantization_per_token,
)
from wrinklefree.quantization.lambda_warmup import (
    LambdaWarmup,
    get_current_lambda,
    get_global_lambda_warmup,
    set_global_lambda_warmup,
)
from wrinklefree.quantization.ste import detach_quantize, ste_quantize
from wrinklefree.quantization.weight_quant import (
    compute_weight_scale,
    ternary_weight_quantization,
    ternary_weight_quantization_no_scale,
)

__all__ = [
    # STE
    "ste_quantize",
    "detach_quantize",
    # Ternary weight quantization
    "ternary_weight_quantization",
    "ternary_weight_quantization_no_scale",
    "compute_weight_scale",
    # Activation quantization
    "activation_quantization_per_token",
    "activation_quantization_per_tensor",
    "activation_quantization_absmean",
    # Lambda warmup
    "LambdaWarmup",
    "get_current_lambda",
    "get_global_lambda_warmup",
    "set_global_lambda_warmup",
]
