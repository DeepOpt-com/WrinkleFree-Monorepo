"""Native BitNet kernels for high-performance CPU inference."""

import os
import sys
from pathlib import Path

# Try to import the compiled extension
try:
    import bitnet_native
    NATIVE_AVAILABLE = True
except ImportError:
    NATIVE_AVAILABLE = False
    bitnet_native = None


def is_available() -> bool:
    """Check if native kernels are available."""
    return NATIVE_AVAILABLE


def build_native():
    """Build the native extension if not available."""
    if NATIVE_AVAILABLE:
        print("Native extension already available")
        return True

    import subprocess
    native_dir = Path(__file__).parent

    print(f"Building native extension in {native_dir}")

    result = subprocess.run(
        [sys.executable, 'setup.py', 'build_ext', '--inplace'],
        cwd=native_dir,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Build failed:\n{result.stderr}")
        return False

    print("Build successful")
    return True


def gemv(weights, input, scale):
    """
    Fused BitNet GEMV: dequantize + matmul in one operation.

    Args:
        weights: [N, K/4] uint8 packed ternary weights
        input: [K] or [batch, K] float32 activations
        scale: float weight scale factor

    Returns:
        [N] or [batch, N] float32 output
    """
    if not NATIVE_AVAILABLE:
        raise RuntimeError("Native extension not available. Run build_native() first.")

    return bitnet_native.gemv(weights, input, scale)


def benchmark(weights, input, scale, iterations=100):
    """Benchmark different GEMV implementations."""
    if not NATIVE_AVAILABLE:
        raise RuntimeError("Native extension not available.")

    return bitnet_native.benchmark(weights, input, scale, iterations)
