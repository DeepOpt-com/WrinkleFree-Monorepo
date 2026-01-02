"""EXPERIMENTAL: FP8-accelerated BitLinear layer.

WARNING: This module is experimental and not production-ready.
APIs may change without notice.

Provides FP8 acceleration for BitLinear using TorchAO (DeepSeek-V3 style).
"""

from wrinklefree._experimental.fp8.fp8_bitlinear import (
    FP8BitLinear,
    convert_bitlinear_to_fp8,
)

__all__ = [
    "FP8BitLinear",
    "convert_bitlinear_to_fp8",
]
