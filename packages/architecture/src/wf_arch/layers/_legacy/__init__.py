"""Legacy layer implementations (deprecated).

These modules are kept for backward compatibility. Please use the new
implementations instead:

- BitLinearLRC -> Use LoRAAdapter wrapper instead
  Old: model = convert_bitlinear_to_lrc(model)
  New: model = add_lora_to_model(model, LoRAConfig())
"""

from wf_arch.layers._legacy.bitlinear_lrc import (
    BitLinearLRC,
    QLRCConfig,
    QuantizedLinearSTE,
    convert_bitlinear_to_lrc,
    freeze_model_except_lrc,
    get_lrc_stats,
)

__all__ = [
    "BitLinearLRC",
    "QLRCConfig",
    "QuantizedLinearSTE",
    "convert_bitlinear_to_lrc",
    "freeze_model_except_lrc",
    "get_lrc_stats",
]
