"""BitNet Architecture Library - 1.58-bit quantized components.

This package provides the core building blocks for BitNet models:
- BitLinear: Ternary weight quantization with 8-bit activation quantization
- SubLN: Sub-Layer Normalization for stable BitNet training
- LambdaWarmup: Gradual quantization schedule
- Conversion utilities: Convert standard models to BitNet on-the-fly
"""

from bitnet_arch.layers import (
    BitLinear,
    BitLinearNoActivationQuant,
    BitLinearLRC,
    SubLN,
    RMSNorm,
    convert_linear_to_bitlinear,
    convert_bitlinear_to_lrc,
    freeze_model_except_lrc,
    get_lrc_stats,
)
from bitnet_arch.quantization import (
    LambdaWarmup,
    get_global_lambda_warmup,
    set_global_lambda_warmup,
    get_current_lambda,
)
from bitnet_arch.conversion import (
    convert_model_to_bitnet,
    is_bitnet_model,
    auto_convert_if_needed,
    run_stage1,
)

__version__ = "0.1.0"

__all__ = [
    # Layers
    "BitLinear",
    "BitLinearNoActivationQuant",
    "BitLinearLRC",
    "SubLN",
    "RMSNorm",
    "convert_linear_to_bitlinear",
    # LRC (Low-Rank Correction)
    "convert_bitlinear_to_lrc",
    "freeze_model_except_lrc",
    "get_lrc_stats",
    # Quantization
    "LambdaWarmup",
    "get_global_lambda_warmup",
    "set_global_lambda_warmup",
    "get_current_lambda",
    # Conversion
    "convert_model_to_bitnet",
    "is_bitnet_model",
    "auto_convert_if_needed",
    "run_stage1",
]
