#!/usr/bin/env python3
"""Local native BitNet kernel benchmark.

Run: python benchmark/local_native_kernel.py
"""

import subprocess
import os
import sys
import time
import tempfile
import json

KERNEL_CPP = r'''
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>
#include <cmath>
#include <cstdint>

namespace py = pybind11;

// Scalar kernel (correct, compiler will auto-vectorize with -O3)
float bitnet_dot_scalar(int n, const uint8_t* packed, const int8_t* act) {
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
        float dot = bitnet_dot_scalar(in_features, w_ptr + i * packed_in, a_ptr);
        r(i) = dot * scale;
    }

    return result;
}

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
            float dot = bitnet_dot_scalar(in_features, w_ptr + i * packed_in, a_ptr + b * in_features);
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
    m.def("gemv", &bitnet_gemv);
    m.def("gemm", &bitnet_gemm);
    m.def("num_threads", &get_num_threads);
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


def main():
    import numpy as np

    print("=" * 60)
    print("Local Native BitNet Kernel Benchmark")
    print("=" * 60)

    # Build in temp directory
    build_dir = tempfile.mkdtemp(prefix="bitnet_build_")
    print(f"\nBuild directory: {build_dir}")

    with open(f"{build_dir}/bitnet_native.cpp", "w") as f:
        f.write(KERNEL_CPP)

    with open(f"{build_dir}/setup.py", "w") as f:
        f.write(SETUP_PY)

    print("\n[1/3] Building native kernel...")
    result = subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=build_dir,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Build failed:\n{result.stderr}")
        return 1

    print("Build successful!")

    sys.path.insert(0, build_dir)
    import bitnet_native

    print(f"OpenMP threads: {bitnet_native.num_threads()}")

    # Helper functions
    def pack_weights(weights):
        out_f, in_f = weights.shape
        packed = np.zeros((out_f, in_f // 4), dtype=np.uint8)
        for i in range(in_f // 4):
            for j in range(4):
                w = (weights[:, i * 4 + j].astype(np.int32) + 1).clip(0, 2)
                packed[:, i] |= (w.astype(np.uint8) << (j * 2))
        return packed

    def unpack_weights(packed):
        out_f = packed.shape[0]
        in_f = packed.shape[1] * 4
        weights = np.zeros((out_f, in_f), dtype=np.float32)
        for i in range(packed.shape[1]):
            for j in range(4):
                w_enc = (packed[:, i] >> (j * 2)) & 0x03
                weights[:, i * 4 + j] = w_enc.astype(np.float32) - 1.0
        return weights

    def gemv_python(packed, act, scale):
        weights = unpack_weights(packed)
        return np.dot(weights, act.astype(np.float32)) * scale

    # ===== Correctness test =====
    print("\n[2/3] Correctness Test...")

    for shape in [(128, 256), (1024, 1024), (2048, 2048)]:
        out_f, in_f = shape
        weights = np.random.randint(-1, 2, (out_f, in_f)).astype(np.float32)
        packed = pack_weights(weights)
        act = np.random.randn(in_f).astype(np.float32)
        act_i8 = np.clip(act * 10, -128, 127).astype(np.int8)

        out_native = bitnet_native.gemv(packed, act_i8, 1.0)
        out_python = gemv_python(packed, act_i8, 1.0)

        cosine = np.dot(out_native, out_python) / (np.linalg.norm(out_native) * np.linalg.norm(out_python))
        status = "PASS" if cosine > 0.9999 else "FAIL"
        print(f"  [{status}] {out_f}x{in_f}: cosine={cosine:.6f}")

    # ===== Performance benchmark =====
    print("\n[3/3] Performance Benchmark...")

    # GEMV
    out_f, in_f = 2048, 2048
    weights = np.random.randint(-1, 2, (out_f, in_f)).astype(np.float32)
    packed = pack_weights(weights)
    act = np.random.randn(in_f).astype(np.float32)
    act_i8 = np.clip(act * 10, -128, 127).astype(np.int8)

    # Warmup
    for _ in range(10):
        _ = bitnet_native.gemv(packed, act_i8, 1.0)

    # Native timing
    native_times = []
    for _ in range(100):
        start = time.perf_counter()
        _ = bitnet_native.gemv(packed, act_i8, 1.0)
        native_times.append(time.perf_counter() - start)

    # Python timing
    python_times = []
    for _ in range(10):
        start = time.perf_counter()
        _ = gemv_python(packed, act_i8, 1.0)
        python_times.append(time.perf_counter() - start)

    native_ms = np.mean(native_times) * 1000
    python_ms = np.mean(python_times) * 1000
    speedup = python_ms / native_ms

    print(f"  GEMV 2048x2048: native={native_ms:.3f}ms, python={python_ms:.1f}ms, speedup={speedup:.1f}x")

    # GEMM with different batch sizes
    for batch in [1, 8, 32, 64, 128]:
        act_batch = np.random.randn(batch, in_f).astype(np.float32)
        act_batch_i8 = np.clip(act_batch * 10, -128, 127).astype(np.int8)

        # Warmup
        for _ in range(5):
            _ = bitnet_native.gemm(packed, act_batch_i8, 1.0)

        # Timing
        times = []
        for _ in range(50):
            start = time.perf_counter()
            _ = bitnet_native.gemm(packed, act_batch_i8, 1.0)
            times.append(time.perf_counter() - start)

        avg_ms = np.mean(times) * 1000
        throughput = batch / np.mean(times)

        print(f"  GEMM batch={batch:3d}: {avg_ms:.2f}ms, {throughput:,.0f} tok/s")

    # ===== Summary =====
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  CPU threads: {bitnet_native.num_threads()}")
    print(f"  GEMV speedup: {speedup:.1f}x vs Python")
    print(f"  Max throughput: {throughput:,.0f} tok/s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
