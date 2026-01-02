"""Native C++ BitNet kernel with OpenMP parallelization.

DEPRECATED: This module builds kernels at runtime and is superseded by sgl-kernel.

Use ``sgl_kernel.quantization`` instead::

    from sgl_kernel.quantization import bitnet_gemv, bitnet_gemm

The sgl-kernel package provides pre-built SIMD-optimized kernels (AVX2/AVX512)
that are faster and don't require runtime compilation.

This module is kept for backward compatibility and testing. It builds and wraps
native GEMV/GEMM kernels for 1.58-bit inference using pybind11 and OpenMP.
"""

import warnings

warnings.warn(
    "wrinklefree_inference.kernels.native is deprecated. "
    "Use sgl_kernel.quantization instead for better performance.",
    DeprecationWarning,
    stacklevel=2,
)

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

# C++ kernel source - scalar with OpenMP parallelization
KERNEL_CPP = r'''
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>
#include <cstdint>

namespace py = pybind11;

// Dot product for packed ternary weights
// Packing: byte[k] = w[4k] | w[4k+1]<<2 | w[4k+2]<<4 | w[4k+3]<<6
// Encoding: 0->-1, 1->0, 2->+1
float bitnet_dot(int n, const uint8_t* packed, const int8_t* act) {
    int32_t sum = 0;
    for (int i = 0; i < n / 4; i++) {
        uint8_t byte = packed[i];
        int8_t w0 = ((byte >> 0) & 3) - 1;
        int8_t w1 = ((byte >> 2) & 3) - 1;
        int8_t w2 = ((byte >> 4) & 3) - 1;
        int8_t w3 = ((byte >> 6) & 3) - 1;
        sum += w0 * (int32_t)act[i * 4 + 0];
        sum += w1 * (int32_t)act[i * 4 + 1];
        sum += w2 * (int32_t)act[i * 4 + 2];
        sum += w3 * (int32_t)act[i * 4 + 3];
    }
    return (float)sum;
}

// GEMV: output[out_features] = packed_weights[out_features, in_features/4] @ activations[in_features]
py::array_t<float> bitnet_gemv(
    py::array_t<uint8_t> packed_weights,
    py::array_t<int8_t> activations,
    float scale
) {
    auto w = packed_weights.unchecked<2>();
    auto a = activations.unchecked<1>();
    int out_features = w.shape(0);
    int packed_in = w.shape(1);
    int in_features = packed_in * 4;

    auto result = py::array_t<float>(out_features);
    auto r = result.mutable_unchecked<1>();
    const uint8_t* w_ptr = w.data(0, 0);
    const int8_t* a_ptr = a.data(0);

    #pragma omp parallel for
    for (int i = 0; i < out_features; i++) {
        r(i) = bitnet_dot(in_features, w_ptr + i * packed_in, a_ptr) * scale;
    }
    return result;
}

// GEMM: output[batch, out_features] = activations[batch, in_features] @ packed_weights.T
py::array_t<float> bitnet_gemm(
    py::array_t<uint8_t> packed_weights,
    py::array_t<int8_t> activations,
    float scale
) {
    auto w = packed_weights.unchecked<2>();
    auto a = activations.unchecked<2>();
    int out_features = w.shape(0);
    int packed_in = w.shape(1);
    int in_features = packed_in * 4;
    int batch_size = a.shape(0);

    auto result = py::array_t<float>({batch_size, out_features});
    auto r = result.mutable_unchecked<2>();
    const uint8_t* w_ptr = w.data(0, 0);
    const int8_t* a_ptr = a.data(0, 0);

    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < out_features; i++) {
            float dot = bitnet_dot(in_features, w_ptr + i * packed_in, a_ptr + b * in_features);
            r(b, i) = dot * scale;
        }
    }
    return result;
}

int get_num_threads() {
    int n = 0;
    #pragma omp parallel
    {
        #pragma omp single
        n = omp_get_num_threads();
    }
    return n;
}

PYBIND11_MODULE(bitnet_native, m) {
    m.def("gemv", &bitnet_gemv, "BitNet GEMV with packed ternary weights");
    m.def("gemm", &bitnet_gemm, "BitNet GEMM with packed ternary weights");
    m.def("num_threads", &get_num_threads, "Get OpenMP thread count");
}
'''

SETUP_PY = '''
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "bitnet_native",
        ["bitnet_native.cpp"],
        extra_compile_args=["-O3", "-march=native", "-fopenmp", "-ffast-math", "-funroll-loops"],
        extra_link_args=["-fopenmp"],
    ),
]

setup(
    name="bitnet_native",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
'''

_kernel_module = None
_build_dir = None


def build_kernel(force: bool = False):
    """Build and import the native kernel module.

    Args:
        force: If True, rebuild even if already built.

    Returns:
        The bitnet_native module with gemv, gemm, num_threads functions.
    """
    global _kernel_module, _build_dir

    if _kernel_module is not None and not force:
        return _kernel_module

    _build_dir = tempfile.mkdtemp(prefix="bitnet_kernel_")

    with open(f"{_build_dir}/bitnet_native.cpp", "w") as f:
        f.write(KERNEL_CPP)
    with open(f"{_build_dir}/setup.py", "w") as f:
        f.write(SETUP_PY)

    result = subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=_build_dir,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Kernel build failed: {result.stderr}")

    sys.path.insert(0, _build_dir)
    import bitnet_native
    _kernel_module = bitnet_native

    return _kernel_module


def get_kernel():
    """Get the kernel module, building if necessary."""
    if _kernel_module is None:
        return build_kernel()
    return _kernel_module


def pack_weights(weights: np.ndarray) -> np.ndarray:
    """Pack float32 ternary weights to uint8 format.

    Input: weights[out_features, in_features] with values in {-1, 0, +1}
    Output: packed[out_features, in_features // 4] as uint8

    Packing: byte[k] = w[4k] | w[4k+1]<<2 | w[4k+2]<<4 | w[4k+3]<<6
    Encoding: -1->0, 0->1, +1->2
    """
    out_f, in_f = weights.shape
    assert in_f % 4 == 0, f"in_features must be divisible by 4, got {in_f}"

    packed = np.zeros((out_f, in_f // 4), dtype=np.uint8)
    for i in range(in_f // 4):
        for j in range(4):
            w = (weights[:, i * 4 + j].astype(np.int32) + 1).clip(0, 2)
            packed[:, i] |= (w.astype(np.uint8) << (j * 2))
    return packed


def unpack_weights(packed: np.ndarray) -> np.ndarray:
    """Unpack uint8 weights back to float32 ternary.

    Input: packed[out_features, in_features // 4] as uint8
    Output: weights[out_features, in_features] with values in {-1, 0, +1}
    """
    out_f = packed.shape[0]
    in_f = packed.shape[1] * 4

    weights = np.zeros((out_f, in_f), dtype=np.float32)
    for i in range(packed.shape[1]):
        for j in range(4):
            w_enc = (packed[:, i] >> (j * 2)) & 0x03
            weights[:, i * 4 + j] = w_enc.astype(np.float32) - 1.0
    return weights


def repack_hf_weights(packed_hf: np.ndarray) -> np.ndarray:
    """Convert HuggingFace packed weights to our kernel format.

    HF format: packed_hf[out_features // 4, in_features] - packing along output dim
    Our format: packed[out_features, in_features // 4] - packing along input dim

    Args:
        packed_hf: uint8 array in HuggingFace format

    Returns:
        uint8 array in our kernel format
    """
    out_packed, in_features = packed_hf.shape
    out_features = out_packed * 4

    # First unpack HF format (packing along output dim)
    unpacked = np.zeros((out_features, in_features), dtype=np.float32)
    packed_np = packed_hf.astype(np.int32)
    for i in range(4):
        bits = (packed_np >> (i * 2)) & 0x03
        unpacked[i::4, :] = bits - 1.0  # 0->-1, 1->0, 2->+1

    # Now repack in our format (packing along input dim)
    return pack_weights(unpacked)


def quantize_activations(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Quantize float32 activations to int8.

    Args:
        x: Float32 activations

    Returns:
        Tuple of (int8 activations, scale factor)
    """
    max_val = np.abs(x).max()
    scale = max_val / 127.0 if max_val > 1e-6 else 1.0
    quantized = np.clip(x / scale, -128, 127).astype(np.int8)
    return quantized, scale
