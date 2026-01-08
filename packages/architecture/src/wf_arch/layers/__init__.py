"""BitNet layer components."""

from wf_arch.layers.bitlinear import (
    BitLinear,
    BitLinearNoActivationQuant,
    convert_linear_to_bitlinear,
)
from wf_arch.layers.bitlinear_salient import (
    BitLinearSalient,
    SalientConfig,
    convert_bitlinear_to_salient,
    get_salient_stats,
)
from wf_arch.layers.salient_calibration import (
    SalientCalibrator,
    calibrate_salient_columns,
)
from wf_arch.layers.subln import SubLN, RMSNorm

# New LoRA adapter (composable wrapper pattern)
from wf_arch.layers.lora_adapter import (
    LoRAAdapter,
    LoRAConfig,
    QuantizedLinearSTE,
    add_lora_to_model,
    freeze_base_model,
    remove_lora_from_model,
    merge_lora_weights,
    get_lora_stats,
    remap_legacy_checkpoint,
)

# Legacy LRC (backward compatibility - use LoRAAdapter instead)
from wf_arch.layers._legacy import (
    BitLinearLRC,
    QLRCConfig,
    convert_bitlinear_to_lrc,
    freeze_model_except_lrc,
    get_lrc_stats,
)

__all__ = [
    # Base layers
    "BitLinear",
    "BitLinearNoActivationQuant",
    "BitLinearSalient",
    "SalientConfig",
    "SalientCalibrator",
    "SubLN",
    "RMSNorm",
    # Conversion utilities
    "convert_linear_to_bitlinear",
    "convert_bitlinear_to_salient",
    "get_salient_stats",
    "calibrate_salient_columns",
    # New LoRA adapter (recommended)
    "LoRAAdapter",
    "LoRAConfig",
    "QuantizedLinearSTE",
    "add_lora_to_model",
    "freeze_base_model",
    "remove_lora_from_model",
    "merge_lora_weights",
    "get_lora_stats",
    "remap_legacy_checkpoint",
    # Legacy LRC (backward compatibility)
    "BitLinearLRC",
    "QLRCConfig",
    "convert_bitlinear_to_lrc",
    "freeze_model_except_lrc",
    "get_lrc_stats",
]
