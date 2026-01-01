"""Model conversion utilities for BitNet."""

from bitnet_arch.conversion.convert import (
    convert_model_to_bitnet,
    is_bitnet_model,
    auto_convert_if_needed,
    run_stage1,
    insert_subln_before_projection,
    convert_attention_layer,
    convert_mlp_layer,
)

__all__ = [
    "convert_model_to_bitnet",
    "is_bitnet_model",
    "auto_convert_if_needed",
    "run_stage1",
    "insert_subln_before_projection",
    "convert_attention_layer",
    "convert_mlp_layer",
]
