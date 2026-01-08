"""BitNet Architecture Library - 1.58-bit quantized components.

This package provides the core building blocks for BitNet models:
- BitLinear: Ternary weight quantization with 8-bit activation quantization
- SubLN: Sub-Layer Normalization for stable BitNet training
- LambdaWarmup: Gradual quantization schedule
- Conversion utilities: Convert standard models to BitNet on-the-fly
"""

from wf_arch.layers import (
    # Base layers
    BitLinear,
    BitLinearNoActivationQuant,
    BitLinearSalient,
    SalientConfig,
    SalientCalibrator,
    SubLN,
    RMSNorm,
    convert_linear_to_bitlinear,
    convert_bitlinear_to_salient,
    get_salient_stats,
    calibrate_salient_columns,
    # New LoRA adapter (recommended)
    LoRAAdapter,
    LoRAConfig,
    add_lora_to_model,
    freeze_base_model,
    remove_lora_from_model,
    merge_lora_weights,
    get_lora_stats,
    remap_legacy_checkpoint,
    # Legacy LRC (backward compatibility)
    BitLinearLRC,
    QLRCConfig,
    convert_bitlinear_to_lrc,
    freeze_model_except_lrc,
    get_lrc_stats,
)
from wf_arch.quantization import (
    LambdaWarmup,
    get_global_lambda_warmup,
    set_global_lambda_warmup,
    get_current_lambda,
)
from wf_arch.conversion import (
    convert_model_to_bitnet,
    is_bitnet_model,
    auto_convert_if_needed,
    run_stage1,
)

__version__ = "0.1.0"

__all__ = [
    # Base Layers
    "BitLinear",
    "BitLinearNoActivationQuant",
    "BitLinearSalient",
    "SalientConfig",
    "SalientCalibrator",
    "SubLN",
    "RMSNorm",
    "convert_linear_to_bitlinear",
    # Salient Columns (AWQ-style)
    "convert_bitlinear_to_salient",
    "get_salient_stats",
    "calibrate_salient_columns",
    # LoRA Adapter (recommended for low-rank correction)
    "LoRAAdapter",
    "LoRAConfig",
    "add_lora_to_model",
    "freeze_base_model",
    "remove_lora_from_model",
    "merge_lora_weights",
    "get_lora_stats",
    "remap_legacy_checkpoint",
    # Legacy LRC (backward compatibility - use LoRAAdapter instead)
    "BitLinearLRC",
    "QLRCConfig",
    "convert_bitlinear_to_lrc",
    "freeze_model_except_lrc",
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
