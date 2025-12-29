#!/usr/bin/env python3
"""Test if the bug is in SIMD kernels or Python code.

Temporarily disable _BITNET_KERNEL_AVAILABLE and test inference.
"""

import sys
import os

# Add the sglang path
sys.path.insert(0, "/home/lev/code/WrinkleFree/WrinkleFree-Inference-Engine/extern/sglang-bitnet/python")

# Import BitNet model module and patch the kernel availability
import sglang.srt.models.bitnet as bitnet_module

print(f"Current _BITNET_KERNEL_AVAILABLE: {bitnet_module._BITNET_KERNEL_AVAILABLE}")

# Let's just check if we can import the functions
try:
    from sgl_kernel.quantization import bitnet_gemm, bitnet_gemv, quantize_activations_i8
    print("Kernels available: True")
except ImportError as e:
    print(f"Kernels available: False ({e})")

# The actual test would need to restart the server with kernels disabled
# For now, let's just verify the kernel availability
print("\nTo test with Python fallback:")
print("1. Edit bitnet.py and set _BITNET_KERNEL_AVAILABLE = False")
print("2. Restart the server")
print("3. Run the debug_repetition.py script")
