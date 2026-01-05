"""SGLang backend integration for BitNet 1.58-bit CPU inference.

This module provides utilities for integrating BitNet models with the SGLang
serving framework, including quantization config and sparse attention support.

Components:
    BitNetConfig: Configuration for BitNet quantization parameters
    BITNET_QUANT_TYPES: Available quantization type constants

See Also:
    - bitnet_quantization.py: Core quantization utilities
    - sparse_attention.py: Sparse attention pattern support
    - activation_sparsity.py: Activation sparsity utilities
"""

from wf_infer.sglang_backend.bitnet_quantization import (
    BITNET_QUANT_TYPES,
    BitNetConfig,
)

__all__ = ["BitNetConfig", "BITNET_QUANT_TYPES"]
