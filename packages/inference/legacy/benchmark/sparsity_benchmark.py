"""Benchmark activation sparsity configurations for BitNet inference.

Measures:
- Throughput (tok/s) at various sparsity levels
- Latency (p50/p99) per forward pass
- Quality (cosine similarity vs dense baseline)
- Actual measured sparsity

Usage:
    uv run python benchmark/sparsity_benchmark.py
    uv run python benchmark/sparsity_benchmark.py --iterations 10
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wrinklefree_inference.sglang_backend.activation_sparsity import (
    ActivationSparsityConfig,
    SparsityMode,
    apply_sparsity,
    get_default_config,
    get_qsparse_config,
    get_conservative_config,
    get_adaptive_config,
    measure_sparsity,
)
from wrinklefree_inference.sglang_backend.bitnet_quantization import (
    BitNetLinearMethod,
    quantize_to_bitnet,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark configuration."""
    name: str
    sparsity_mode: str
    target_sparsity: float
    actual_sparsity: float
    throughput_ops_per_sec: float
    latency_ms_p50: float
    latency_ms_p99: float
    cosine_similarity: float
    max_diff: float
    num_iterations: int
    batch_size: int
    hidden_dim: int

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "sparsity_mode": self.sparsity_mode,
            "target_sparsity": self.target_sparsity,
            "actual_sparsity": self.actual_sparsity,
            "throughput_ops_per_sec": self.throughput_ops_per_sec,
            "latency_ms_p50": self.latency_ms_p50,
            "latency_ms_p99": self.latency_ms_p99,
            "cosine_similarity": self.cosine_similarity,
            "max_diff": self.max_diff,
            "num_iterations": self.num_iterations,
            "batch_size": self.batch_size,
            "hidden_dim": self.hidden_dim,
        }


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two tensors."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return (torch.dot(a_flat, b_flat) / (torch.norm(a_flat) * torch.norm(b_flat) + 1e-8)).item()


def run_single_benchmark(
    method: BitNetLinearMethod,
    packed_weight: torch.Tensor,
    scale: float,
    out_features: int,
    in_features: int,
    batch_size: int,
    sparsity_config: ActivationSparsityConfig,
    num_warmup: int = 10,
    num_iterations: int = 100,
    baseline_output: Optional[torch.Tensor] = None,
    fixed_input: Optional[torch.Tensor] = None,
) -> BenchmarkResult:
    """Run benchmark for a single sparsity configuration."""

    # Use fixed input if provided, otherwise generate random
    x = fixed_input if fixed_input is not None else torch.randn(batch_size, in_features, dtype=torch.float32)

    # Warmup
    for _ in range(num_warmup):
        _ = method.apply(packed_weight, scale, x, out_features, in_features, sparsity_config=sparsity_config)

    # Benchmark
    latencies = []
    sparsities = []

    for _ in range(num_iterations):
        start = time.perf_counter()
        out = method.apply(packed_weight, scale, x, out_features, in_features, sparsity_config=sparsity_config)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms
        sparsities.append(method.get_last_sparsity())

    # Compute statistics
    latencies = np.array(latencies)
    p50 = np.percentile(latencies, 50)
    p99 = np.percentile(latencies, 99)
    ops_per_sec = num_iterations / (latencies.sum() / 1000)
    avg_sparsity = np.mean(sparsities)

    # Compute quality metrics vs baseline
    if baseline_output is not None:
        cos_sim = cosine_similarity(out, baseline_output)
        max_diff = (out - baseline_output).abs().max().item()
    else:
        cos_sim = 1.0
        max_diff = 0.0

    # Determine target sparsity from config
    if sparsity_config.mode == SparsityMode.TOP_K:
        target = 1.0 - sparsity_config.top_k_ratio
    elif sparsity_config.mode == SparsityMode.ADAPTIVE:
        target = 1.0 - (sparsity_config.adaptive_min_ratio + sparsity_config.adaptive_max_ratio) / 2
    else:
        target = 0.0

    return BenchmarkResult(
        name=f"{sparsity_config.mode.value}_{sparsity_config.top_k_ratio if sparsity_config.mode == SparsityMode.TOP_K else 'default'}",
        sparsity_mode=sparsity_config.mode.value,
        target_sparsity=target,
        actual_sparsity=avg_sparsity,
        throughput_ops_per_sec=ops_per_sec,
        latency_ms_p50=p50,
        latency_ms_p99=p99,
        cosine_similarity=cos_sim,
        max_diff=max_diff,
        num_iterations=num_iterations,
        batch_size=batch_size,
        hidden_dim=in_features,
    )


def run_sparsity_sweep(
    hidden_dim: int = 4096,
    batch_size: int = 1,
    num_iterations: int = 100,
    num_warmup: int = 10,
) -> List[BenchmarkResult]:
    """Run a sweep of sparsity configurations."""

    logger.info(f"Running sparsity benchmark: hidden_dim={hidden_dim}, batch_size={batch_size}")

    # Create synthetic weights
    weight = torch.randn(hidden_dim, hidden_dim, dtype=torch.float32)
    packed_weight, scale = quantize_to_bitnet(weight)

    # Create method with default (no sparsity)
    method = BitNetLinearMethod()

    # Create fixed input for fair comparison
    x = torch.randn(batch_size, hidden_dim, dtype=torch.float32)

    # Get baseline output (no sparsity) with the fixed input
    baseline_config = get_default_config()
    baseline_output = method.apply(packed_weight, scale, x, hidden_dim, hidden_dim, sparsity_config=baseline_config)

    results = []

    # Test configurations
    configs = [
        ("dense", get_default_config()),
        ("conservative_30", get_conservative_config()),
        ("qsparse_60", get_qsparse_config()),
        ("adaptive", get_adaptive_config()),
    ]

    # Add sweep of top_k ratios
    for ratio in [0.5, 0.4, 0.35, 0.3, 0.25, 0.2]:
        cfg = ActivationSparsityConfig(
            enabled=True,
            mode=SparsityMode.TOP_K,
            top_k_ratio=ratio,
            track_stats=True,
        )
        sparsity_pct = int((1 - ratio) * 100)
        configs.append((f"top_k_{sparsity_pct}pct", cfg))

    for name, config in configs:
        logger.info(f"  Benchmarking: {name}")
        result = run_single_benchmark(
            method=method,
            packed_weight=packed_weight,
            scale=scale,
            out_features=hidden_dim,
            in_features=hidden_dim,
            batch_size=batch_size,
            sparsity_config=config,
            num_warmup=num_warmup,
            num_iterations=num_iterations,
            baseline_output=baseline_output if config.enabled else None,
            fixed_input=x,  # Use fixed input for fair comparison
        )
        result.name = name
        results.append(result)

        quality_status = "PASS" if result.cosine_similarity > 0.99 else "WARN" if result.cosine_similarity > 0.95 else "FAIL"
        logger.info(
            f"    {name}: sparsity={result.actual_sparsity:.1%}, "
            f"throughput={result.throughput_ops_per_sec:.1f} ops/s, "
            f"cosine={result.cosine_similarity:.6f} [{quality_status}]"
        )

    return results


def run_iteration_benchmark(
    num_outer_iterations: int = 10,
    hidden_dim: int = 4096,
    batch_size: int = 1,
    num_inner_iterations: int = 50,
) -> Dict:
    """Run multiple iterations of the benchmark and find optimal config."""

    logger.info(f"Running {num_outer_iterations} iterations of sparsity benchmark")

    all_results = []

    for i in range(num_outer_iterations):
        logger.info(f"\n=== Iteration {i+1}/{num_outer_iterations} ===")
        results = run_sparsity_sweep(
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            num_iterations=num_inner_iterations,
        )
        all_results.extend(results)

    # Aggregate results by config name
    aggregated = {}
    for r in all_results:
        if r.name not in aggregated:
            aggregated[r.name] = {
                "throughputs": [],
                "latencies_p50": [],
                "latencies_p99": [],
                "cosine_sims": [],
                "sparsities": [],
            }
        aggregated[r.name]["throughputs"].append(r.throughput_ops_per_sec)
        aggregated[r.name]["latencies_p50"].append(r.latency_ms_p50)
        aggregated[r.name]["latencies_p99"].append(r.latency_ms_p99)
        aggregated[r.name]["cosine_sims"].append(r.cosine_similarity)
        aggregated[r.name]["sparsities"].append(r.actual_sparsity)

    # Compute summary statistics
    summary = {}
    for name, data in aggregated.items():
        summary[name] = {
            "throughput_mean": np.mean(data["throughputs"]),
            "throughput_std": np.std(data["throughputs"]),
            "latency_p50_mean": np.mean(data["latencies_p50"]),
            "latency_p99_mean": np.mean(data["latencies_p99"]),
            "cosine_sim_mean": np.mean(data["cosine_sims"]),
            "cosine_sim_min": np.min(data["cosine_sims"]),
            "sparsity_mean": np.mean(data["sparsities"]),
        }

    return {
        "num_iterations": num_outer_iterations,
        "hidden_dim": hidden_dim,
        "batch_size": batch_size,
        "summary": summary,
    }


def find_optimal_config(summary: Dict) -> Tuple[str, Dict]:
    """Find the optimal sparsity configuration.

    Criteria:
    1. Cosine similarity > 0.99 (quality threshold)
    2. Maximize throughput
    """
    candidates = []

    for name, stats in summary["summary"].items():
        if stats["cosine_sim_min"] > 0.99:
            candidates.append((name, stats))

    if not candidates:
        # Relax quality threshold
        for name, stats in summary["summary"].items():
            if stats["cosine_sim_min"] > 0.95:
                candidates.append((name, stats))

    if not candidates:
        return "dense", summary["summary"]["dense"]

    # Sort by throughput (higher is better)
    candidates.sort(key=lambda x: x[1]["throughput_mean"], reverse=True)

    return candidates[0]


def print_results_table(summary: Dict):
    """Print results as a formatted table."""
    print("\n" + "="*100)
    print(f"SPARSITY BENCHMARK RESULTS (hidden_dim={summary['hidden_dim']}, batch_size={summary['batch_size']})")
    print("="*100)
    print(f"{'Config':<20} {'Sparsity':>10} {'Throughput':>12} {'Latency P50':>12} {'Cosine Sim':>12} {'Status':>8}")
    print("-"*100)

    baseline_throughput = summary["summary"]["dense"]["throughput_mean"]

    for name, stats in sorted(summary["summary"].items(), key=lambda x: x[1]["throughput_mean"], reverse=True):
        speedup = stats["throughput_mean"] / baseline_throughput
        quality = "PASS" if stats["cosine_sim_min"] > 0.99 else "WARN" if stats["cosine_sim_min"] > 0.95 else "FAIL"
        print(
            f"{name:<20} "
            f"{stats['sparsity_mean']:>9.1%} "
            f"{stats['throughput_mean']:>10.1f}x{speedup:>3.2f} "
            f"{stats['latency_p50_mean']:>10.3f}ms "
            f"{stats['cosine_sim_mean']:>11.6f} "
            f"{quality:>8}"
        )

    print("-"*100)

    # Find and print optimal
    optimal_name, optimal_stats = find_optimal_config(summary)
    speedup = optimal_stats["throughput_mean"] / baseline_throughput
    print(f"\nOPTIMAL CONFIG: {optimal_name}")
    print(f"  Sparsity: {optimal_stats['sparsity_mean']:.1%}")
    print(f"  Speedup vs dense: {speedup:.2f}x")
    print(f"  Quality (min cosine): {optimal_stats['cosine_sim_min']:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark activation sparsity configurations")
    parser.add_argument("--iterations", type=int, default=10, help="Number of outer iterations")
    parser.add_argument("--hidden-dim", type=int, default=4096, help="Hidden dimension")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--inner-iterations", type=int, default=50, help="Inner iterations per config")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()

    summary = run_iteration_benchmark(
        num_outer_iterations=args.iterations,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        num_inner_iterations=args.inner_iterations,
    )

    print_results_table(summary)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Results saved to {args.output}")

    return summary


if __name__ == "__main__":
    main()
