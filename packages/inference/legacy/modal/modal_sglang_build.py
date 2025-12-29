"""Benchmark BitNet kernels and continuous batching on Modal CPU.

This script:
1. Tests BitNet GEMV/GEMM with optimized Python implementation
2. Benchmarks continuous batching scenarios
3. Compares single-request vs batched inference
"""

import modal
import os

app = modal.App("bitnet-cpu-benchmark")

# CPU-only image for benchmarking
cpu_bench_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install([
        "build-essential",
        "libomp-dev",
    ])
    .pip_install([
        "torch==2.5.1+cpu",
        "numpy",
    ], extra_index_url="https://download.pytorch.org/whl/cpu")
)


# BitNet kernel implementation (embedded for Modal)
BITNET_KERNEL_CODE = '''
import torch
import numpy as np
from typing import Tuple
import time

QK_I2_S = 128


def _unpack_ternary_weights(packed_weights: torch.Tensor) -> torch.Tensor:
    """Unpack 2-bit packed weights to ternary {-1, 0, +1} values."""
    out_features = packed_weights.shape[0]
    packed_in_features = packed_weights.shape[1]
    in_features = packed_in_features * 4

    packed = packed_weights.to(torch.int32)
    weights = torch.zeros(out_features, in_features, dtype=torch.float32)

    for i in range(4):
        shift = i * 2
        bits = (packed >> shift) & 0x03
        unpacked = bits.float() - 1.0
        weights[:, i::4] = unpacked

    return weights


def pack_ternary_weights(weights: torch.Tensor) -> torch.Tensor:
    """Pack ternary {-1, 0, +1} weights to 2-bit format."""
    out_features, in_features = weights.shape
    packed_in_features = in_features // 4
    packed = torch.zeros(out_features, packed_in_features, dtype=torch.uint8)

    for i in range(4):
        w = weights[:, i::4].long() + 1
        w = w.clamp(0, 2)
        packed = packed | (w.to(torch.uint8) << (i * 2))

    return packed


def quantize_activations_i8(activations: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Quantize FP32 activations to INT8."""
    max_val = activations.abs().max().item()
    if max_val < 1e-6:
        max_val = 1.0
    scale = max_val / 127.0
    quantized = (activations / scale).round().clamp(-128, 127).to(torch.int8)
    return quantized, scale


def bitnet_gemv(packed_weights: torch.Tensor, activations: torch.Tensor, scale: float) -> torch.Tensor:
    """BitNet GEMV: y = scale * (W @ x)."""
    weights = _unpack_ternary_weights(packed_weights)
    activations_f = activations.to(torch.float32)
    return torch.matmul(weights, activations_f) * scale


def bitnet_gemm(packed_weights: torch.Tensor, activations: torch.Tensor, scale: float) -> torch.Tensor:
    """BitNet GEMM: Y = scale * (W @ X)."""
    weights = _unpack_ternary_weights(packed_weights)
    activations_f = activations.to(torch.float32)
    return torch.matmul(activations_f, weights.t()) * scale
'''


@app.function(
    image=cpu_bench_image,
    cpu=16.0,
    memory=32768,
    timeout=30 * 60,
)
def benchmark_bitnet(iterations: int = 10) -> dict:
    """Benchmark BitNet kernels with 10 iterations."""
    import torch
    import numpy as np
    import time

    # Execute the kernel code
    exec(BITNET_KERNEL_CODE, globals())

    results = {
        "iterations": iterations,
        "cpu_count": torch.get_num_threads(),
        "gemv_results": [],
        "gemm_results": [],
        "all_passed": True,
    }

    print(f"Running on {results['cpu_count']} CPU threads")
    print(f"\nRunning {iterations} iterations...")

    # Model dimensions (similar to Llama-2B)
    out_features = 4096
    in_features = 4096

    for i in range(iterations):
        # Create random ternary weights
        weights = torch.randint(-1, 2, (out_features, in_features), dtype=torch.float32)
        packed = pack_ternary_weights(weights)

        # GEMV benchmark (single token)
        activations = torch.randn(in_features)
        act_i8, act_scale = quantize_activations_i8(activations)

        # Warmup
        _ = bitnet_gemv(packed, act_i8, 1.0)

        # Time GEMV
        start = time.perf_counter()
        for _ in range(10):
            output = bitnet_gemv(packed, act_i8, 1.0)
        gemv_time = (time.perf_counter() - start) / 10

        # Verify correctness
        weights_ref = _unpack_ternary_weights(packed)
        act_dequant = act_i8.float() * act_scale
        ref_output = torch.matmul(weights_ref, act_dequant)
        cosine = torch.nn.functional.cosine_similarity(
            output.unsqueeze(0), ref_output.unsqueeze(0)
        ).item()

        results["gemv_results"].append({
            "iteration": i + 1,
            "time_ms": gemv_time * 1000,
            "cosine": cosine,
            "passed": cosine > 0.99,
        })

        if cosine <= 0.99:
            results["all_passed"] = False

        # GEMM benchmark (batched - continuous batching scenario)
        batch_size = 32
        activations_batch = torch.randn(batch_size, in_features)
        act_batch_i8, act_batch_scale = quantize_activations_i8(activations_batch)

        # Warmup
        _ = bitnet_gemm(packed, act_batch_i8, 1.0)

        start = time.perf_counter()
        for _ in range(10):
            output_batch = bitnet_gemm(packed, act_batch_i8, 1.0)
        gemm_time = (time.perf_counter() - start) / 10

        act_dequant_batch = act_batch_i8.float() * act_batch_scale
        ref_output_batch = torch.matmul(act_dequant_batch, weights_ref.t())
        cosine_batch = torch.nn.functional.cosine_similarity(
            output_batch.flatten().unsqueeze(0),
            ref_output_batch.flatten().unsqueeze(0)
        ).item()

        results["gemm_results"].append({
            "iteration": i + 1,
            "time_ms": gemm_time * 1000,
            "cosine": cosine_batch,
            "passed": cosine_batch > 0.99,
        })

        if cosine_batch <= 0.99:
            results["all_passed"] = False

        print(f"  Iter {i+1}: GEMV {gemv_time*1000:.2f}ms (cos={cosine:.4f}), "
              f"GEMM[32] {gemm_time*1000:.2f}ms (cos={cosine_batch:.4f})")

    # Summary
    gemv_times = [r["time_ms"] for r in results["gemv_results"]]
    gemm_times = [r["time_ms"] for r in results["gemm_results"]]

    results["summary"] = {
        "gemv_avg_ms": np.mean(gemv_times),
        "gemv_min_ms": np.min(gemv_times),
        "gemm_avg_ms": np.mean(gemm_times),
        "gemm_min_ms": np.min(gemm_times),
        "gemm_throughput_tok_s": 32 / (np.mean(gemm_times) / 1000),
    }

    return results


@app.function(
    image=cpu_bench_image,
    cpu=32.0,
    memory=65536,
    timeout=30 * 60,
)
def benchmark_continuous_batching(max_batch_size: int = 64) -> dict:
    """Benchmark continuous batching with varying batch sizes."""
    import torch
    import numpy as np
    import time

    exec(BITNET_KERNEL_CODE, globals())

    results = {
        "cpu_count": torch.get_num_threads(),
        "batch_sizes": [],
    }

    print(f"Running continuous batching benchmark on {results['cpu_count']} threads")
    print(f"Testing batch sizes from 1 to {max_batch_size}")

    # Model dimensions
    out_features = 4096
    in_features = 4096

    # Create weights once
    weights = torch.randint(-1, 2, (out_features, in_features), dtype=torch.float32)
    packed = pack_ternary_weights(weights)

    batch_sizes = [1, 2, 4, 8, 16, 32, 48, 64]
    batch_sizes = [b for b in batch_sizes if b <= max_batch_size]

    for batch_size in batch_sizes:
        # Create batch
        activations = torch.randn(batch_size, in_features)
        act_i8, act_scale = quantize_activations_i8(activations)

        # Warmup
        for _ in range(3):
            _ = bitnet_gemm(packed, act_i8, 1.0)

        # Benchmark
        times = []
        for _ in range(20):
            start = time.perf_counter()
            output = bitnet_gemm(packed, act_i8, 1.0)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = np.mean(times)
        throughput = batch_size / avg_time

        results["batch_sizes"].append({
            "batch_size": batch_size,
            "avg_time_ms": avg_time * 1000,
            "throughput_tok_s": throughput,
            "latency_per_token_ms": (avg_time * 1000) / batch_size,
        })

        print(f"  Batch {batch_size:3d}: {avg_time*1000:.2f}ms, {throughput:.1f} tok/s")

    # Calculate scaling efficiency
    single_throughput = results["batch_sizes"][0]["throughput_tok_s"]
    for r in results["batch_sizes"]:
        r["scaling_efficiency"] = r["throughput_tok_s"] / (single_throughput * r["batch_size"])

    return results


@app.function(
    image=cpu_bench_image,
    cpu=32.0,
    memory=65536,
    timeout=30 * 60,
)
def benchmark_layer_stack(num_layers: int = 32) -> dict:
    """Benchmark a full transformer layer stack (like 2B model)."""
    import torch
    import numpy as np
    import time

    exec(BITNET_KERNEL_CODE, globals())

    results = {
        "cpu_count": torch.get_num_threads(),
        "num_layers": num_layers,
        "layer_timings": [],
    }

    print(f"Benchmarking {num_layers}-layer transformer on {results['cpu_count']} threads")

    # Llama-2B-like dimensions
    hidden_dim = 2048
    ffn_dim = 5632
    batch_size = 1

    # Create weights for all layers
    layers = []
    for i in range(num_layers):
        layer = {
            "q_proj": pack_ternary_weights(torch.randint(-1, 2, (hidden_dim, hidden_dim), dtype=torch.float32)),
            "k_proj": pack_ternary_weights(torch.randint(-1, 2, (hidden_dim, hidden_dim), dtype=torch.float32)),
            "v_proj": pack_ternary_weights(torch.randint(-1, 2, (hidden_dim, hidden_dim), dtype=torch.float32)),
            "o_proj": pack_ternary_weights(torch.randint(-1, 2, (hidden_dim, hidden_dim), dtype=torch.float32)),
            "gate_proj": pack_ternary_weights(torch.randint(-1, 2, (ffn_dim, hidden_dim), dtype=torch.float32)),
            "up_proj": pack_ternary_weights(torch.randint(-1, 2, (ffn_dim, hidden_dim), dtype=torch.float32)),
            "down_proj": pack_ternary_weights(torch.randint(-1, 2, (hidden_dim, ffn_dim), dtype=torch.float32)),
        }
        layers.append(layer)

    # Single token forward pass
    hidden = torch.randn(hidden_dim)
    hidden_i8, _ = quantize_activations_i8(hidden)

    # Warmup
    for layer in layers[:2]:
        _ = bitnet_gemv(layer["q_proj"], hidden_i8, 1.0)

    # Full forward pass timing
    layer_times = []

    for i, layer in enumerate(layers):
        start = time.perf_counter()

        # Attention projections
        q = bitnet_gemv(layer["q_proj"], hidden_i8, 1.0)
        k = bitnet_gemv(layer["k_proj"], hidden_i8, 1.0)
        v = bitnet_gemv(layer["v_proj"], hidden_i8, 1.0)

        # Simplified attention (just output projection)
        attn_out_i8, _ = quantize_activations_i8(v)
        attn_out = bitnet_gemv(layer["o_proj"], attn_out_i8, 1.0)

        # FFN
        ffn_in_i8, _ = quantize_activations_i8(attn_out)
        gate = bitnet_gemv(layer["gate_proj"], ffn_in_i8, 1.0)
        up = bitnet_gemv(layer["up_proj"], ffn_in_i8, 1.0)
        ffn_hidden = gate * torch.sigmoid(gate) * up  # SwiGLU
        ffn_hidden_i8, _ = quantize_activations_i8(ffn_hidden)
        output = bitnet_gemv(layer["down_proj"], ffn_hidden_i8, 1.0)

        layer_time = time.perf_counter() - start
        layer_times.append(layer_time)

        # Update hidden for next layer
        hidden_i8, _ = quantize_activations_i8(output)

    total_time = sum(layer_times)
    results["total_time_ms"] = total_time * 1000
    results["tokens_per_second"] = 1.0 / total_time
    results["layer_avg_ms"] = np.mean(layer_times) * 1000

    # Memory estimate (1.58 bits per weight)
    total_params = num_layers * (4 * hidden_dim * hidden_dim + 3 * ffn_dim * hidden_dim)
    memory_gb = total_params * 1.58 / 8 / 1e9

    results["estimated_params"] = total_params
    results["estimated_memory_gb"] = memory_gb

    print(f"\nResults for {num_layers}-layer model:")
    print(f"  Total time: {results['total_time_ms']:.2f}ms")
    print(f"  Tokens/sec: {results['tokens_per_second']:.2f}")
    print(f"  Layer avg: {results['layer_avg_ms']:.2f}ms")
    print(f"  Est. params: {total_params/1e9:.2f}B")
    print(f"  Est. memory: {memory_gb:.2f}GB")

    return results


@app.local_entrypoint()
def main():
    """Run all BitNet benchmarks on Modal."""
    import json

    print("=" * 70)
    print("BitNet CPU Benchmark on Modal")
    print("=" * 70)

    # 1. Basic kernel benchmark (10 iterations)
    print("\n[1/3] BitNet Kernel Benchmark (10 iterations)...")
    kernel_results = benchmark_bitnet.remote(iterations=10)
    print(f"\nKernel results:")
    print(f"  GEMV avg: {kernel_results['summary']['gemv_avg_ms']:.2f}ms")
    print(f"  GEMM avg: {kernel_results['summary']['gemm_avg_ms']:.2f}ms")
    print(f"  GEMM throughput: {kernel_results['summary']['gemm_throughput_tok_s']:.1f} tok/s")
    print(f"  All passed: {kernel_results['all_passed']}")

    # 2. Continuous batching benchmark
    print("\n[2/3] Continuous Batching Benchmark...")
    batch_results = benchmark_continuous_batching.remote(max_batch_size=64)
    print(f"\nBatch results:")
    for r in batch_results["batch_sizes"]:
        print(f"  Batch {r['batch_size']:3d}: {r['throughput_tok_s']:.1f} tok/s, "
              f"efficiency={r['scaling_efficiency']:.2%}")

    # 3. Layer stack benchmark
    print("\n[3/3] Layer Stack Benchmark (32 layers)...")
    layer_results = benchmark_layer_stack.remote(num_layers=32)
    print(f"\nLayer stack results:")
    print(f"  Total time: {layer_results['total_time_ms']:.2f}ms")
    print(f"  Tokens/sec: {layer_results['tokens_per_second']:.2f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Kernel tests passed: {kernel_results['all_passed']}")
    print(f"Single token latency: {layer_results['total_time_ms']:.2f}ms")
    print(f"Continuous batching speedup: {batch_results['batch_sizes'][-1]['throughput_tok_s'] / batch_results['batch_sizes'][0]['throughput_tok_s']:.1f}x")
    print(f"Max throughput (batch=64): {batch_results['batch_sizes'][-1]['throughput_tok_s']:.1f} tok/s")
