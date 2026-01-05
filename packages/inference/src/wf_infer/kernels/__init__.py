"""Native SIMD kernels for BitNet 1.58-bit inference.

This module provides optimized CPU kernels for BitNet inference using
SIMD instructions (AVX2/AVX512). The kernels handle packed ternary weight
multiplication with quantized activations.

Components:
    NATIVE_KERNELS_AVAILABLE: Whether sgl-kernel is installed
    BitNetLinearNative: Drop-in replacement for BitLinear using native kernels
    patch_model_with_native_kernels: Convert a model to use native kernels
    benchmark_kernels: Performance benchmarking utilities

Performance:
    - AVX512: ~29 tok/s on GCP C3D instances
    - AVX2: ~20 tok/s on standard CPUs

Note:
    The native.py module is deprecated. Use sgl_kernel.quantization directly
    for new code.
"""

from .bitnet_patch import (
    NATIVE_KERNELS_AVAILABLE,
    BitNetLinearNative,
    benchmark_kernels,
    patch_model_with_native_kernels,
)

__all__ = [
    "NATIVE_KERNELS_AVAILABLE",
    "BitNetLinearNative",
    "patch_model_with_native_kernels",
    "benchmark_kernels",
]
