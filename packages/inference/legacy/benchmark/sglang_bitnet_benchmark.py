#!/usr/bin/env python3
"""SGLang BitNet Benchmark Suite.

Benchmarks BitNet 1.58-bit inference performance:
- Quantization speed
- GEMV/GEMM throughput
- Memory efficiency
- Comparison with baseline

Usage:
    uv run python benchmark/sglang_bitnet_benchmark.py
    uv run python benchmark/sglang_bitnet_benchmark.py --quick  # Fast validation
"""

import argparse
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
import sys
from pathlib import Path

import torch


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    name: str
    duration_ms: float
    throughput: float  # ops/sec or tokens/sec
    memory_mb: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    results: List[BenchmarkResult] = field(default_factory=list)
    system_info: Dict = field(default_factory=dict)

    def add(self, result: BenchmarkResult):
        self.results.append(result)

    def summary(self) -> str:
        lines = ["=" * 60]
        lines.append("SGLang BitNet Benchmark Results")
        lines.append("=" * 60)

        for r in self.results:
            lines.append(f"\n{r.name}:")
            lines.append(f"  Duration: {r.duration_ms:.2f} ms")
            lines.append(f"  Throughput: {r.throughput:.2f}")
            lines.append(f"  Memory: {r.memory_mb:.2f} MB")
            if r.metadata:
                for k, v in r.metadata.items():
                    lines.append(f"  {k}: {v}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps({
            "system_info": self.system_info,
            "results": [
                {
                    "name": r.name,
                    "duration_ms": r.duration_ms,
                    "throughput": r.throughput,
                    "memory_mb": r.memory_mb,
                    "metadata": r.metadata,
                }
                for r in self.results
            ]
        }, indent=2)


def get_system_info() -> Dict:
    """Collect system information."""
    import platform

    info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
    }

    # CPU info
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    info["cpu_model"] = line.split(":")[1].strip()
                    break
    except:
        pass

    # Check for AVX support
    try:
        import subprocess
        result = subprocess.run(
            ["grep", "-o", "avx[^ ]*", "/proc/cpuinfo"],
            capture_output=True, text=True
        )
        avx_flags = list(set(result.stdout.strip().split()))
        info["avx_support"] = avx_flags
    except:
        info["avx_support"] = []

    return info


def benchmark_quantization(sizes: List[tuple], warmup: int = 3, iterations: int = 10) -> List[BenchmarkResult]:
    """Benchmark quantization speed for various matrix sizes."""
    from wrinklefree_inference.sglang_backend.bitnet_quantization import quantize_to_bitnet

    results = []

    for out_features, in_features in sizes:
        weights = torch.randn(out_features, in_features)
        total_params = out_features * in_features

        # Warmup
        for _ in range(warmup):
            quantize_to_bitnet(weights)

        # Benchmark
        times = []
        for _ in range(iterations):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.perf_counter()
            quantize_to_bitnet(weights)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            times.append(time.perf_counter() - start)

        avg_time = sum(times) / len(times)
        throughput = total_params / avg_time / 1e6  # M params/sec

        # Memory
        packed, _ = quantize_to_bitnet(weights)
        memory_mb = packed.numel() * packed.element_size() / (1024 * 1024)

        results.append(BenchmarkResult(
            name=f"Quantize {out_features}x{in_features}",
            duration_ms=avg_time * 1000,
            throughput=throughput,
            memory_mb=memory_mb,
            metadata={
                "params": f"{total_params / 1e6:.2f}M",
                "compression_ratio": f"{weights.numel() * 4 / packed.numel():.1f}x",
            }
        ))

    return results


def benchmark_gemv(sizes: List[tuple], warmup: int = 5, iterations: int = 20) -> List[BenchmarkResult]:
    """Benchmark GEMV (batch=1) speed."""
    from wrinklefree_inference.sglang_backend.bitnet_quantization import (
        quantize_to_bitnet,
        BitNetLinearMethod,
    )

    results = []
    method = BitNetLinearMethod()

    for out_features, in_features in sizes:
        weights = torch.randn(out_features, in_features)
        packed, scale = quantize_to_bitnet(weights)
        x = torch.randn(1, in_features)

        # Warmup
        for _ in range(warmup):
            method.apply(packed, scale, x, out_features, in_features)

        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            method.apply(packed, scale, x, out_features, in_features)
            times.append(time.perf_counter() - start)

        avg_time = sum(times) / len(times)
        # Throughput: operations per second
        ops = 2 * out_features * in_features  # MAC ops
        throughput = ops / avg_time / 1e9  # GOPS

        results.append(BenchmarkResult(
            name=f"GEMV {out_features}x{in_features}",
            duration_ms=avg_time * 1000,
            throughput=throughput,
            memory_mb=packed.numel() * packed.element_size() / (1024 * 1024),
            metadata={
                "GOPS": f"{throughput:.2f}",
                "batch_size": 1,
            }
        ))

    return results


def benchmark_gemm(sizes: List[tuple], batch_sizes: List[int], warmup: int = 5, iterations: int = 10) -> List[BenchmarkResult]:
    """Benchmark batched GEMM speed."""
    from wrinklefree_inference.sglang_backend.bitnet_quantization import (
        quantize_to_bitnet,
        BitNetLinearMethod,
    )

    results = []
    method = BitNetLinearMethod()

    for out_features, in_features in sizes:
        weights = torch.randn(out_features, in_features)
        packed, scale = quantize_to_bitnet(weights)

        for batch_size in batch_sizes:
            x = torch.randn(batch_size, in_features)

            # Warmup
            for _ in range(warmup):
                method.apply(packed, scale, x, out_features, in_features)

            # Benchmark
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                method.apply(packed, scale, x, out_features, in_features)
                times.append(time.perf_counter() - start)

            avg_time = sum(times) / len(times)
            ops = 2 * batch_size * out_features * in_features
            throughput = ops / avg_time / 1e9  # GOPS

            results.append(BenchmarkResult(
                name=f"GEMM {out_features}x{in_features} batch={batch_size}",
                duration_ms=avg_time * 1000,
                throughput=throughput,
                memory_mb=packed.numel() * packed.element_size() / (1024 * 1024),
                metadata={
                    "GOPS": f"{throughput:.2f}",
                    "batch_size": batch_size,
                }
            ))

    return results


def run_quick_benchmark() -> BenchmarkSuite:
    """Run quick validation benchmark."""
    suite = BenchmarkSuite()
    suite.system_info = get_system_info()

    print("Running quick benchmark...")

    # Small sizes for quick validation
    sizes = [(512, 512), (2048, 2048)]
    batch_sizes = [1, 8]

    suite.results.extend(benchmark_quantization(sizes, warmup=1, iterations=3))
    suite.results.extend(benchmark_gemv(sizes, warmup=2, iterations=5))
    suite.results.extend(benchmark_gemm([(2048, 2048)], batch_sizes, warmup=2, iterations=5))

    return suite


def run_full_benchmark() -> BenchmarkSuite:
    """Run comprehensive benchmark suite."""
    suite = BenchmarkSuite()
    suite.system_info = get_system_info()

    print("Running full benchmark suite...")

    # Realistic model sizes (BitNet 2B dimensions)
    sizes = [
        (2560, 2560),   # Hidden size
        (2560, 6912),   # FFN intermediate
        (6912, 2560),   # FFN output
        (4096, 4096),   # Larger model
        (4096, 11008),  # Llama-style FFN
    ]
    batch_sizes = [1, 4, 8, 16, 32, 64]

    print("  Quantization benchmarks...")
    suite.results.extend(benchmark_quantization(sizes))

    print("  GEMV benchmarks...")
    suite.results.extend(benchmark_gemv(sizes))

    print("  GEMM benchmarks...")
    suite.results.extend(benchmark_gemm(sizes[:2], batch_sizes))

    return suite


def main():
    parser = argparse.ArgumentParser(description="SGLang BitNet Benchmark")
    parser.add_argument("--quick", action="store_true", help="Run quick validation")
    parser.add_argument("--output", "-o", type=str, help="Output JSON file")
    args = parser.parse_args()

    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root / "src"))

    if args.quick:
        suite = run_quick_benchmark()
    else:
        suite = run_full_benchmark()

    print(suite.summary())

    if args.output:
        with open(args.output, "w") as f:
            f.write(suite.to_json())
        print(f"\nResults saved to {args.output}")

    # Quick validation check
    print("\nValidation:")
    for r in suite.results:
        if r.throughput > 0:
            print(f"  [OK] {r.name}")
        else:
            print(f"  [FAIL] {r.name}")


if __name__ == "__main__":
    main()
