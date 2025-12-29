#!/usr/bin/env python3
"""Micro-benchmark for BitNet kernels with detailed profiling output.

Run on GCP: python scripts/benchmark_kernels.py
Profile: perf record -g -- python scripts/benchmark_kernels.py && perf report
"""

import time
import torch
import numpy as np
from sgl_kernel.quantization import (
    bitnet_gemv,
    bitnet_gemm,
    bitnet_quantize_activations as quantize_activations_i8,
    bitnet_check_kernel_available as check_kernel_available,
    BITNET_BLOCK_SIZE as QK_I2_S,
)

def pack_ternary_weights(weights: torch.Tensor) -> torch.Tensor:
    """Pack ternary weights {-1, 0, +1} to 2-bit packed format."""
    out_features, in_features = weights.shape
    packed = torch.zeros(out_features, in_features // 4, dtype=torch.uint8)

    for i in range(4):
        # Map: -1 -> 0, 0 -> 1, +1 -> 2
        vals = (weights[:, i::4] + 1).to(torch.uint8)
        packed |= (vals << (i * 2))

    return packed


def benchmark_gemv(in_features: int, out_features: int, warmup: int = 10, iters: int = 100) -> dict:
    """Benchmark GEMV kernel."""
    # Generate random ternary weights
    weights = torch.randint(-1, 2, (out_features, in_features), dtype=torch.float32)
    packed = pack_ternary_weights(weights)

    # Generate random activations
    activations_fp = torch.randn(in_features)
    activations, act_scale = quantize_activations_i8(activations_fp)

    scale = 0.01  # Weight scale

    # Warmup
    for _ in range(warmup):
        _ = bitnet_gemv(packed, activations, scale)

    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    for _ in range(iters):
        _ = bitnet_gemv(packed, activations, scale)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - start

    # Metrics
    ms_per_call = (elapsed / iters) * 1000
    flops = 2 * in_features * out_features  # multiply-add
    gflops = (flops * iters / elapsed) / 1e9

    # Memory bandwidth (weights + activations + output)
    weight_bytes = packed.numel()  # uint8
    act_bytes = activations.numel()  # int8
    out_bytes = out_features * 4  # float32
    total_bytes = weight_bytes + act_bytes + out_bytes
    bandwidth_gb = (total_bytes * iters / elapsed) / 1e9

    return {
        "ms": ms_per_call,
        "gflops": gflops,
        "bandwidth_gb": bandwidth_gb,
        "iters": iters,
    }


def benchmark_gemm(in_features: int, out_features: int, batch: int, warmup: int = 10, iters: int = 100) -> dict:
    """Benchmark GEMM kernel."""
    # Generate random ternary weights
    weights = torch.randint(-1, 2, (out_features, in_features), dtype=torch.float32)
    packed = pack_ternary_weights(weights)

    # Generate random batched activations
    activations_fp = torch.randn(batch, in_features)
    activations, act_scale = quantize_activations_i8(activations_fp)

    scale = 0.01

    # Warmup
    for _ in range(warmup):
        _ = bitnet_gemm(packed, activations, scale)

    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        _ = bitnet_gemm(packed, activations, scale)
    elapsed = time.perf_counter() - start

    ms_per_call = (elapsed / iters) * 1000
    flops = 2 * batch * in_features * out_features
    gflops = (flops * iters / elapsed) / 1e9

    weight_bytes = packed.numel()
    act_bytes = activations.numel()
    out_bytes = batch * out_features * 4
    total_bytes = weight_bytes + act_bytes + out_bytes
    bandwidth_gb = (total_bytes * iters / elapsed) / 1e9

    return {
        "ms": ms_per_call,
        "gflops": gflops,
        "bandwidth_gb": bandwidth_gb,
        "iters": iters,
    }


def main():
    print("=" * 70)
    print("BitNet Kernel Micro-Benchmark")
    print("=" * 70)

    # Check kernel availability
    if not check_kernel_available():
        print("ERROR: BitNet kernels not available!")
        return

    print(f"Kernel available: True")
    print(f"Block size (QK_I2_S): {QK_I2_S}")
    print()

    # Model dimensions for BitNet-b1.58-2B-4T
    # Hidden: 2560, Intermediate: 6912
    model_configs = [
        # (name, in_features, out_features)
        ("QKV Proj (2560->2560)", 2560, 2560),
        ("Up Proj (2560->6912)", 2560, 6912),
        ("Down Proj (6912->2560)", 6912, 2560),
        ("Gate Proj (2560->6912)", 2560, 6912),
    ]

    batch_sizes = [1, 4, 8, 16, 32]

    # GEMV benchmarks (batch=1)
    print("=" * 70)
    print("GEMV Benchmarks (batch=1)")
    print("=" * 70)
    print(f"{'Layer':<30} {'ms':>10} {'GFLOPS':>10} {'GB/s':>10}")
    print("-" * 70)

    gemv_results = {}
    for name, in_feat, out_feat in model_configs:
        result = benchmark_gemv(in_feat, out_feat)
        gemv_results[name] = result
        print(f"{name:<30} {result['ms']:>10.3f} {result['gflops']:>10.2f} {result['bandwidth_gb']:>10.2f}")

    print()

    # GEMM benchmarks (various batch sizes)
    print("=" * 70)
    print("GEMM Benchmarks (various batch sizes)")
    print("=" * 70)

    # Test with Down Proj as representative layer
    in_feat, out_feat = 6912, 2560
    print(f"Layer: Down Proj ({in_feat}->{out_feat})")
    print(f"{'Batch':<10} {'ms':>10} {'GFLOPS':>10} {'GB/s':>10}")
    print("-" * 70)

    for batch in batch_sizes:
        result = benchmark_gemm(in_feat, out_feat, batch)
        print(f"{batch:<10} {result['ms']:>10.3f} {result['gflops']:>10.2f} {result['bandwidth_gb']:>10.2f}")

    print()

    # Stress test for profiling
    print("=" * 70)
    print("Stress Test (for perf profiling)")
    print("=" * 70)

    # Run many iterations for profiling
    stress_iters = 1000
    weights = torch.randint(-1, 2, (2560, 2560), dtype=torch.float32)
    packed = pack_ternary_weights(weights)
    activations_fp = torch.randn(2560)
    activations, _ = quantize_activations_i8(activations_fp)

    print(f"Running {stress_iters} GEMV iterations (2560x2560)...")
    start = time.perf_counter()
    for _ in range(stress_iters):
        _ = bitnet_gemv(packed, activations, 0.01)
    elapsed = time.perf_counter() - start

    print(f"Total time: {elapsed:.3f}s")
    print(f"Avg per call: {elapsed/stress_iters*1000:.3f}ms")
    print(f"Throughput: {stress_iters/elapsed:.1f} calls/sec")

    print()
    print("=" * 70)
    print("Baseline Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
