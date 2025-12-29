"""KV cache benchmark for long context windows.

Tests:
1. Memory usage at various context lengths
2. Latency for cache update and retrieval
3. Quantization quality (FP8, INT8 vs BF16)
4. Attention correctness with cached KV

Usage:
    python benchmark/kv_cache_bench.py
    python benchmark/kv_cache_bench.py --context-lengths 1024 4096 8192 16384
    python benchmark/kv_cache_bench.py --iterations 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wrinklefree_inference.kv_cache.kv_cache import (
    KVCache,
    KVCacheConfig,
    KVCacheDtype,
    attention_with_kv_cache,
    compute_kv_cache_memory,
)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    context_length: int
    dtype: str
    memory_mb: float
    update_latency_us: float  # microseconds per token
    get_latency_us: float
    attention_latency_us: float
    cosine_sim_vs_bf16: float  # Quality vs BF16 baseline
    mse_vs_bf16: float


def benchmark_kv_cache(
    context_lengths: list[int],
    dtypes: list[KVCacheDtype],
    num_layers: int = 32,
    num_heads: int = 32,
    head_dim: int = 128,
    warmup_iters: int = 10,
    bench_iters: int = 100,
) -> list[BenchmarkResult]:
    """Run KV cache benchmarks.

    Args:
        context_lengths: Context lengths to test
        dtypes: Data types to test
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        head_dim: Dimension per head
        warmup_iters: Warmup iterations
        bench_iters: Benchmark iterations

    Returns:
        List of benchmark results
    """
    results = []

    for ctx_len in context_lengths:
        print(f"\n{'='*60}")
        print(f" Context Length: {ctx_len} tokens")
        print(f"{'='*60}")

        # First run BF16 as baseline for quality comparison
        bf16_cache = None
        bf16_output = None

        for dtype in dtypes:
            config = KVCacheConfig(
                max_seq_len=ctx_len,
                num_layers=num_layers,
                num_heads=num_heads,
                head_dim=head_dim,
                dtype=dtype,
            )

            cache = KVCache(config)

            # Generate random KV data
            batch_size = 1
            seq_len = min(ctx_len, 512)  # Test with subset
            key = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.bfloat16)
            value = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.bfloat16)

            # Warmup
            for _ in range(warmup_iters):
                cache.clear()
                cache.update(0, key, value, 0)
                _ = cache.get(0)

            # Benchmark update
            cache.clear()
            torch.cuda.synchronize() if torch.cuda.is_available() else None

            update_times = []
            for _ in range(bench_iters):
                cache.clear()
                start = time.perf_counter()
                for layer_idx in range(min(4, num_layers)):  # Test 4 layers
                    cache.update(layer_idx, key, value, 0)
                elapsed = time.perf_counter() - start
                update_times.append(elapsed)

            update_latency_us = (sum(update_times) / len(update_times)) * 1e6 / (seq_len * 4)

            # Benchmark get
            get_times = []
            for _ in range(bench_iters):
                start = time.perf_counter()
                for layer_idx in range(min(4, num_layers)):
                    _ = cache.get(layer_idx)
                elapsed = time.perf_counter() - start
                get_times.append(elapsed)

            get_latency_us = (sum(get_times) / len(get_times)) * 1e6 / (seq_len * 4)

            # Benchmark attention
            query = torch.randn(batch_size, num_heads, 1, head_dim, dtype=torch.bfloat16)
            new_key = torch.randn(batch_size, 1, num_heads, head_dim, dtype=torch.bfloat16)
            new_value = torch.randn(batch_size, 1, num_heads, head_dim, dtype=torch.bfloat16)

            cache.clear()
            cache.update(0, key, value, 0)

            attn_times = []
            for _ in range(bench_iters):
                start = time.perf_counter()
                output = attention_with_kv_cache(
                    query, cache, 0, new_key, new_value, seq_len
                )
                elapsed = time.perf_counter() - start
                attn_times.append(elapsed)

            attention_latency_us = (sum(attn_times) / len(attn_times)) * 1e6

            # Quality comparison vs BF16
            if dtype == KVCacheDtype.BF16:
                bf16_cache = cache
                bf16_output = output
                cosine_sim = 1.0
                mse = 0.0
            else:
                # Compare attention output quality
                if bf16_output is not None:
                    # Run same inputs through BF16 cache
                    bf16_cache.clear()
                    bf16_cache.update(0, key, value, 0)
                    ref_output = attention_with_kv_cache(
                        query, bf16_cache, 0, new_key, new_value, seq_len
                    )

                    # Compute quality metrics
                    output_flat = output.flatten().float()
                    ref_flat = ref_output.flatten().float()

                    cosine_sim = F.cosine_similarity(
                        output_flat.unsqueeze(0), ref_flat.unsqueeze(0)
                    ).item()
                    mse = F.mse_loss(output_flat, ref_flat).item()
                else:
                    cosine_sim = 1.0
                    mse = 0.0

            result = BenchmarkResult(
                context_length=ctx_len,
                dtype=dtype.value,
                memory_mb=cache.memory_usage_mb(),
                update_latency_us=update_latency_us,
                get_latency_us=get_latency_us,
                attention_latency_us=attention_latency_us,
                cosine_sim_vs_bf16=cosine_sim,
                mse_vs_bf16=mse,
            )
            results.append(result)

            print(f"\n{dtype.value}:")
            print(f"  Memory:     {result.memory_mb:.1f} MB")
            print(f"  Update:     {result.update_latency_us:.2f} µs/token")
            print(f"  Get:        {result.get_latency_us:.2f} µs/token")
            print(f"  Attention:  {result.attention_latency_us:.2f} µs")
            print(f"  Cosine sim: {result.cosine_sim_vs_bf16:.6f}")
            if mse > 0:
                print(f"  MSE:        {result.mse_vs_bf16:.2e}")

    return results


def benchmark_correctness(
    context_length: int = 4096,
    num_layers: int = 4,
    num_heads: int = 32,
    head_dim: int = 128,
    num_tests: int = 10,
) -> dict[str, Any]:
    """Test correctness of quantized KV cache vs BF16 baseline.

    Args:
        context_length: Context length to test
        num_layers: Number of layers
        num_heads: Number of heads
        head_dim: Head dimension
        num_tests: Number of random tests

    Returns:
        Dict with correctness metrics
    """
    print(f"\n{'='*60}")
    print(" Correctness Test: Quantized vs BF16 Baseline")
    print(f"{'='*60}")

    dtypes_to_test = [
        KVCacheDtype.FP8_E4M3,
        KVCacheDtype.FP8_E5M2,
        KVCacheDtype.INT8,
    ]

    results = {}

    # Create BF16 baseline cache
    bf16_config = KVCacheConfig(
        max_seq_len=context_length,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        dtype=KVCacheDtype.BF16,
    )
    bf16_cache = KVCache(bf16_config)

    for dtype in dtypes_to_test:
        config = KVCacheConfig(
            max_seq_len=context_length,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
        )
        quant_cache = KVCache(config)

        cosine_sims = []
        mses = []

        for _ in range(num_tests):
            # Random data
            seq_len = 512
            key = torch.randn(1, seq_len, num_heads, head_dim, dtype=torch.bfloat16)
            value = torch.randn(1, seq_len, num_heads, head_dim, dtype=torch.bfloat16)
            query = torch.randn(1, num_heads, 1, head_dim, dtype=torch.bfloat16)
            new_key = torch.randn(1, 1, num_heads, head_dim, dtype=torch.bfloat16)
            new_value = torch.randn(1, 1, num_heads, head_dim, dtype=torch.bfloat16)

            # Run through both caches
            bf16_cache.clear()
            quant_cache.clear()

            bf16_cache.update(0, key, value, 0)
            quant_cache.update(0, key, value, 0)

            bf16_output = attention_with_kv_cache(
                query, bf16_cache, 0, new_key, new_value, seq_len
            )
            quant_output = attention_with_kv_cache(
                query, quant_cache, 0, new_key, new_value, seq_len
            )

            # Compare
            bf16_flat = bf16_output.flatten().float()
            quant_flat = quant_output.flatten().float()

            cos_sim = F.cosine_similarity(
                bf16_flat.unsqueeze(0), quant_flat.unsqueeze(0)
            ).item()
            mse = F.mse_loss(bf16_flat, quant_flat).item()

            cosine_sims.append(cos_sim)
            mses.append(mse)

        avg_cos = sum(cosine_sims) / len(cosine_sims)
        avg_mse = sum(mses) / len(mses)
        min_cos = min(cosine_sims)

        results[dtype.value] = {
            "avg_cosine_sim": avg_cos,
            "min_cosine_sim": min_cos,
            "avg_mse": avg_mse,
            "pass": min_cos > 0.99,  # Threshold for acceptable quality
        }

        status = "PASS" if min_cos > 0.99 else "FAIL"
        print(f"\n{dtype.value}:")
        print(f"  Avg cosine sim: {avg_cos:.6f}")
        print(f"  Min cosine sim: {min_cos:.6f}")
        print(f"  Avg MSE:        {avg_mse:.2e}")
        print(f"  Status:         {status}")

    return results


def run_optimization_iteration(
    iteration: int,
    context_length: int = 8192,
    num_layers: int = 32,
) -> dict[str, Any]:
    """Run a single optimization iteration.

    This is called 20 times to iteratively optimize the KV cache.

    Returns:
        Dict with iteration results
    """
    print(f"\n{'='*60}")
    print(f" Optimization Iteration {iteration}/20")
    print(f"{'='*60}")

    # Run benchmarks
    results = benchmark_kv_cache(
        context_lengths=[context_length],
        dtypes=[
            KVCacheDtype.BF16,
            KVCacheDtype.FP8_E4M3,
            KVCacheDtype.INT8,
        ],
        num_layers=num_layers,
        warmup_iters=5,
        bench_iters=50,
    )

    # Run correctness
    correctness = benchmark_correctness(
        context_length=context_length,
        num_layers=min(4, num_layers),
    )

    # Find best config
    best_result = min(results, key=lambda r: r.attention_latency_us)
    best_quality = max(results, key=lambda r: r.cosine_sim_vs_bf16)

    summary = {
        "iteration": iteration,
        "context_length": context_length,
        "results": [
            {
                "dtype": r.dtype,
                "memory_mb": r.memory_mb,
                "attention_us": r.attention_latency_us,
                "cosine_sim": r.cosine_sim_vs_bf16,
            }
            for r in results
        ],
        "best_speed": best_result.dtype,
        "best_quality": best_quality.dtype,
        "correctness_pass": all(c["pass"] for c in correctness.values()),
    }

    print(f"\nSummary:")
    print(f"  Best speed:   {best_result.dtype} ({best_result.attention_latency_us:.2f} µs)")
    print(f"  Best quality: {best_quality.dtype} (cosine={best_quality.cosine_sim_vs_bf16:.6f})")
    print(f"  Correctness:  {'PASS' if summary['correctness_pass'] else 'FAIL'}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="KV cache benchmark")
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=[1024, 4096, 8192],
        help="Context lengths to test",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of optimization iterations (use 20 for full sweep)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=32,
        help="Number of transformer layers",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )

    args = parser.parse_args()

    print("=" * 60)
    print(" KV Cache Benchmark")
    print("=" * 60)
    print(f"Context lengths: {args.context_lengths}")
    print(f"Iterations: {args.iterations}")
    print(f"Layers: {args.num_layers}")

    if args.iterations > 1:
        # Run optimization iterations
        all_results = []
        for i in range(1, args.iterations + 1):
            result = run_optimization_iteration(
                i,
                context_length=max(args.context_lengths),
                num_layers=args.num_layers,
            )
            all_results.append(result)

        # Save results
        if args.output:
            with open(args.output, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"\nResults saved to {args.output}")
    else:
        # Single run with all context lengths
        results = benchmark_kv_cache(
            context_lengths=args.context_lengths,
            dtypes=[
                KVCacheDtype.BF16,
                KVCacheDtype.FP16,
                KVCacheDtype.FP8_E4M3,
                KVCacheDtype.INT8,
            ],
            num_layers=args.num_layers,
        )

        # Correctness test
        correctness = benchmark_correctness()

        # Print summary
        print("\n" + "=" * 60)
        print(" Summary")
        print("=" * 60)
        print("\nMemory savings vs BF16:")
        bf16_mem = {r.context_length: r.memory_mb for r in results if r.dtype == "bfloat16"}
        for r in results:
            if r.dtype != "bfloat16" and r.context_length in bf16_mem:
                savings = (1 - r.memory_mb / bf16_mem[r.context_length]) * 100
                print(f"  {r.dtype} @ {r.context_length}: {savings:.1f}% less memory")

        print("\nQuality check:")
        for dtype, metrics in correctness.items():
            status = "PASS" if metrics["pass"] else "FAIL"
            print(f"  {dtype}: cosine={metrics['avg_cosine_sim']:.6f} [{status}]")


if __name__ == "__main__":
    main()
