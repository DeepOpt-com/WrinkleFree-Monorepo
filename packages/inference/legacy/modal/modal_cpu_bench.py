"""Modal CPU benchmark for BitNet native kernel optimization.

Run 20 optimization iterations on high-throughput CPU instances.

Usage:
    modal run benchmark/modal_cpu_bench.py --iterations 20

    # Single benchmark
    modal run benchmark/modal_cpu_bench.py::benchmark_native

    # Thread sweep
    modal run benchmark/modal_cpu_bench.py --sweep

    # List results
    modal run benchmark/modal_cpu_bench.py --list-results
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import modal

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Modal app
app = modal.App("wrinklefree-cpu-bench")

# Persistent volume for results
results_volume = modal.Volume.from_name(
    "wrinklefree-bench-results", create_if_missing=True
)

# CPU image with all dependencies for native kernel compilation
# Use Image.add_local_dir for mounting local source (Modal 1.0+)
cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git", "curl", "build-essential", "cmake",
        # For AVX512 support detection
        "cpuinfo",
    )
    .pip_install(
        "torch>=2.5.0",
        "numpy>=1.26.0",
        "pybind11>=2.11.0",
        "requests>=2.31.0",  # For wrinklefree_inference client
        "aiohttp>=3.9.0",    # For async client
    )
    .env({
        "OMP_NUM_THREADS": "16",  # Will be adjusted at runtime
    })
    # Add local source code to the image
    .add_local_dir(
        str(PROJECT_ROOT / "src"),
        remote_path="/app/src",
    )
)


@dataclass
class BenchResult:
    """Benchmark result for a single run."""
    cpu_type: str
    num_threads: int
    ms_per_layer: float
    tok_s: float
    memory_bw_pct: float  # Estimated memory bandwidth utilization
    timestamp: str
    config: dict[str, Any]


def estimate_memory_bandwidth(ms_per_layer: float, model_size_gb: float = 0.875) -> float:
    """Estimate memory bandwidth utilization.

    7B model with 1.58-bit = 7B * 1.58 / 8 = 1.38 GB
    But we're benchmarking a layer, so:
    - Layer has ~200M params (4096x4096 + 3x4096x11008)
    - 200M * 1.58 / 8 = 39.5 MB per layer

    Theoretical peak (DDR5-4800): ~77 GB/s per channel, ~307 GB/s for 4 channels
    """
    layer_size_mb = 39.5  # Approximate for 7B transformer layer
    layer_size_gb = layer_size_mb / 1024

    # GB/s = layer_size / (ms/1000)
    achieved_bw = layer_size_gb / (ms_per_layer / 1000)

    # Assume ~200 GB/s theoretical peak for server CPUs
    theoretical_peak = 200.0

    return min(100.0, (achieved_bw / theoretical_peak) * 100)


@app.function(
    image=cpu_image,
    cpu=16.0,  # Request 16 vCPUs
    memory=32768,  # 32GB RAM
    timeout=30 * 60,  # 30 minutes
    volumes={"/results": results_volume},
)
def benchmark_native(
    num_threads: int = 8,
    iterations: int = 50,
) -> dict[str, Any]:
    """Run native kernel benchmark on Modal CPU.

    Args:
        num_threads: Number of OMP threads
        iterations: Number of benchmark iterations

    Returns:
        Benchmark results dict
    """
    import os
    import subprocess
    import sys
    import time
    from pathlib import Path

    # Use source code added to the image
    src_dir = Path("/app/src")
    native_dir = src_dir / "wrinklefree_inference/native"

    # Build native extension
    print("[Modal] Building native kernel...")
    result = subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=native_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Build stderr: {result.stderr}")
        raise RuntimeError(f"Failed to build native kernel: {result.stderr}")

    # Add to path
    sys.path.insert(0, str(native_dir))
    sys.path.insert(0, str(src_dir))

    # Set thread count
    os.environ["OMP_NUM_THREADS"] = str(num_threads)

    # Now run the benchmark
    print(f"[Modal] Running benchmark with {num_threads} threads, {iterations} iterations...")

    import torch
    torch.set_num_threads(num_threads)

    # Import after building
    from wrinklefree_inference.sglang_backend.bitnet_quantization import (
        quantize_to_bitnet, BitNetLinearMethod
    )
    import bitnet_native

    HIDDEN_DIM = 4096
    INTERMEDIATE_DIM = 11008
    NUM_LAYERS = 32

    layer_weights = {
        "q_proj": (HIDDEN_DIM, HIDDEN_DIM),
        "k_proj": (HIDDEN_DIM, HIDDEN_DIM),
        "v_proj": (HIDDEN_DIM, HIDDEN_DIM),
        "o_proj": (HIDDEN_DIM, HIDDEN_DIM),
        "gate_proj": (INTERMEDIATE_DIM, HIDDEN_DIM),
        "up_proj": (INTERMEDIATE_DIM, HIDDEN_DIM),
        "down_proj": (HIDDEN_DIM, INTERMEDIATE_DIM),
    }

    # Quantize weights
    packed_weights = {}
    scales = {}
    for name, (out_dim, in_dim) in layer_weights.items():
        w = torch.randn(out_dim, in_dim)
        packed, scale = quantize_to_bitnet(w)
        packed_weights[name] = packed
        scales[name] = scale

    # Pre-dequantize weights for cached approach
    cached_weights = {}
    for name in packed_weights:
        cached_weights[name] = bitnet_native.dequant(packed_weights[name], scales[name])

    x = torch.randn(1, HIDDEN_DIM, dtype=torch.float32)
    x_f32 = x.float().contiguous()
    x_flat = x_f32.squeeze(0)

    def forward_native_fused():
        """Fused dequant+matmul (original)."""
        q = bitnet_native.gemv(packed_weights["q_proj"], x_f32, scales["q_proj"])
        k = bitnet_native.gemv(packed_weights["k_proj"], x_f32, scales["k_proj"])
        v = bitnet_native.gemv(packed_weights["v_proj"], x_f32, scales["v_proj"])
        o = bitnet_native.gemv(packed_weights["o_proj"], x_f32, scales["o_proj"])
        gate = bitnet_native.gemv(packed_weights["gate_proj"], x_f32, scales["gate_proj"])
        up = bitnet_native.gemv(packed_weights["up_proj"], x_f32, scales["up_proj"])
        hidden = gate * up
        down = bitnet_native.gemv(packed_weights["down_proj"], hidden.contiguous(), scales["down_proj"])
        return down

    def forward_native_cached():
        """Cached weights + torch.mv."""
        q = torch.mv(cached_weights["q_proj"], x_flat)
        k = torch.mv(cached_weights["k_proj"], x_flat)
        v = torch.mv(cached_weights["v_proj"], x_flat)
        o = torch.mv(cached_weights["o_proj"], x_flat)
        gate = torch.mv(cached_weights["gate_proj"], x_flat)
        up = torch.mv(cached_weights["up_proj"], x_flat)
        hidden = gate * up
        down = torch.mv(cached_weights["down_proj"], hidden.contiguous())
        return down

    # Also benchmark Python baseline for comparison
    method = BitNetLinearMethod(compute_dtype=torch.bfloat16)

    def forward_python():
        q = method.apply(packed_weights["q_proj"], scales["q_proj"], x, HIDDEN_DIM, HIDDEN_DIM)
        k = method.apply(packed_weights["k_proj"], scales["k_proj"], x, HIDDEN_DIM, HIDDEN_DIM)
        v = method.apply(packed_weights["v_proj"], scales["v_proj"], x, HIDDEN_DIM, HIDDEN_DIM)
        o = method.apply(packed_weights["o_proj"], scales["o_proj"], x, HIDDEN_DIM, HIDDEN_DIM)
        gate = method.apply(packed_weights["gate_proj"], scales["gate_proj"], x, INTERMEDIATE_DIM, HIDDEN_DIM)
        up = method.apply(packed_weights["up_proj"], scales["up_proj"], x, INTERMEDIATE_DIM, HIDDEN_DIM)
        hidden = gate * up
        down = method.apply(packed_weights["down_proj"], scales["down_proj"], hidden, HIDDEN_DIM, INTERMEDIATE_DIM)
        return down

    # Warmup
    for _ in range(10):
        forward_native_fused()
        forward_native_cached()
        forward_python()

    # Benchmark native fused
    start = time.perf_counter()
    for _ in range(iterations):
        forward_native_fused()
    fused_elapsed = time.perf_counter() - start
    fused_ms = fused_elapsed / iterations * 1000

    # Benchmark native cached
    start = time.perf_counter()
    for _ in range(iterations):
        forward_native_cached()
    cached_elapsed = time.perf_counter() - start
    cached_ms = cached_elapsed / iterations * 1000

    # Benchmark Python
    start = time.perf_counter()
    for _ in range(iterations):
        forward_python()
    python_elapsed = time.perf_counter() - start
    python_ms = python_elapsed / iterations * 1000

    native_ms = fused_ms  # Keep for compatibility

    # Get CPU info
    cpu_info = subprocess.run(
        ["lscpu"], capture_output=True, text=True
    ).stdout
    cpu_model = "unknown"
    for line in cpu_info.split("\n"):
        if "Model name" in line:
            cpu_model = line.split(":")[1].strip()
            break

    fused_model_ms = fused_ms * NUM_LAYERS
    fused_tok_s = 1000 / fused_model_ms

    cached_model_ms = cached_ms * NUM_LAYERS
    cached_tok_s = 1000 / cached_model_ms

    python_model_ms = python_ms * NUM_LAYERS
    python_tok_s = 1000 / python_model_ms

    # Best of fused and cached
    native_ms = min(fused_ms, cached_ms)
    native_model_ms = native_ms * NUM_LAYERS
    native_tok_s = 1000 / native_model_ms

    speedup = python_ms / native_ms
    mem_bw = estimate_memory_bandwidth(cached_ms)  # Cached is memory-bound

    result = {
        "cpu_model": cpu_model,
        "num_threads": num_threads,
        "iterations": iterations,
        "native_fused": {
            "ms_per_layer": fused_ms,
            "ms_per_model": fused_model_ms,
            "tok_s": fused_tok_s,
        },
        "native_cached": {
            "ms_per_layer": cached_ms,
            "ms_per_model": cached_model_ms,
            "tok_s": cached_tok_s,
        },
        "native": {
            "ms_per_layer": native_ms,
            "ms_per_model": native_model_ms,
            "tok_s": native_tok_s,
        },
        "python": {
            "ms_per_layer": python_ms,
            "ms_per_model": python_model_ms,
            "tok_s": python_tok_s,
        },
        "speedup": speedup,
        "memory_bw_pct": mem_bw,
        "timestamp": datetime.now().isoformat(),
    }

    print("\n" + "=" * 60)
    print(f" CPU: {cpu_model}")
    print(f" Threads: {num_threads}")
    print("=" * 60)
    print(f" Fused:   {fused_ms:.2f}ms/layer, {fused_tok_s:.2f} tok/s")
    print(f" Cached:  {cached_ms:.2f}ms/layer, {cached_tok_s:.2f} tok/s")
    print(f" Python:  {python_ms:.2f}ms/layer, {python_tok_s:.2f} tok/s")
    print(f" Best:    {native_ms:.2f}ms/layer, {native_tok_s:.2f} tok/s ({speedup:.2f}x vs Python)")
    print(f" Mem BW:  {mem_bw:.1f}%")
    print("=" * 60)

    # Save result
    results_file = Path("/results/bench_history.jsonl")
    with open(results_file, "a") as f:
        f.write(json.dumps(result) + "\n")
    results_volume.commit()

    return result


@app.function(
    image=cpu_image,
    cpu=32.0,  # Max CPUs for thread sweep
    memory=65536,  # 64GB RAM
    timeout=60 * 60,  # 1 hour
    volumes={"/results": results_volume},
)
def thread_sweep(
    thread_counts: list[int] | None = None,
    iterations: int = 50,
) -> dict[str, Any]:
    """Run thread count sweep to find optimal parallelization.

    Args:
        thread_counts: List of thread counts to test
        iterations: Number of benchmark iterations

    Returns:
        Results for all thread counts
    """
    if thread_counts is None:
        thread_counts = [1, 2, 4, 8, 12, 16, 24, 32]

    import os
    import subprocess
    import sys
    import time
    from pathlib import Path

    # Use source code added to the image
    src_dir = Path("/app/src")
    native_dir = src_dir / "wrinklefree_inference/native"

    # Build native extension
    print("[Modal] Building native kernel...")
    result = subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=native_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Build stderr: {result.stderr}")
        raise RuntimeError(f"Failed to build native kernel: {result.stderr}")

    # Add to path
    sys.path.insert(0, str(native_dir))
    sys.path.insert(0, str(src_dir))

    import torch
    from wrinklefree_inference.sglang_backend.bitnet_quantization import quantize_to_bitnet
    import bitnet_native

    HIDDEN_DIM = 4096
    INTERMEDIATE_DIM = 11008
    NUM_LAYERS = 32

    layer_weights = {
        "q_proj": (HIDDEN_DIM, HIDDEN_DIM),
        "k_proj": (HIDDEN_DIM, HIDDEN_DIM),
        "v_proj": (HIDDEN_DIM, HIDDEN_DIM),
        "o_proj": (HIDDEN_DIM, HIDDEN_DIM),
        "gate_proj": (INTERMEDIATE_DIM, HIDDEN_DIM),
        "up_proj": (INTERMEDIATE_DIM, HIDDEN_DIM),
        "down_proj": (HIDDEN_DIM, INTERMEDIATE_DIM),
    }

    # Quantize weights
    packed_weights = {}
    scales = {}
    for name, (out_dim, in_dim) in layer_weights.items():
        w = torch.randn(out_dim, in_dim)
        packed, scale = quantize_to_bitnet(w)
        packed_weights[name] = packed
        scales[name] = scale

    x = torch.randn(1, HIDDEN_DIM, dtype=torch.float32)
    x_f32 = x.float().contiguous()

    # Get CPU info
    cpu_info = subprocess.run(
        ["lscpu"], capture_output=True, text=True
    ).stdout
    cpu_model = "unknown"
    for line in cpu_info.split("\n"):
        if "Model name" in line:
            cpu_model = line.split(":")[1].strip()
            break

    print(f"\n[Modal] CPU: {cpu_model}")
    print(f"[Modal] Testing threads: {thread_counts}")
    print("\nThreads | ms/layer | tok/s | Speedup")
    print("-" * 45)

    results = []
    baseline_ms = None

    for num_threads in thread_counts:
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        torch.set_num_threads(num_threads)

        def forward():
            q = bitnet_native.gemv(packed_weights["q_proj"], x_f32, scales["q_proj"])
            k = bitnet_native.gemv(packed_weights["k_proj"], x_f32, scales["k_proj"])
            v = bitnet_native.gemv(packed_weights["v_proj"], x_f32, scales["v_proj"])
            o = bitnet_native.gemv(packed_weights["o_proj"], x_f32, scales["o_proj"])
            gate = bitnet_native.gemv(packed_weights["gate_proj"], x_f32, scales["gate_proj"])
            up = bitnet_native.gemv(packed_weights["up_proj"], x_f32, scales["up_proj"])
            hidden = gate * up
            down = bitnet_native.gemv(packed_weights["down_proj"], hidden.contiguous(), scales["down_proj"])
            return down

        # Warmup
        for _ in range(10):
            forward()

        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            forward()
        elapsed = time.perf_counter() - start

        ms_per_layer = elapsed / iterations * 1000
        model_ms = ms_per_layer * NUM_LAYERS
        tok_s = 1000 / model_ms

        if baseline_ms is None:
            baseline_ms = ms_per_layer

        speedup = baseline_ms / ms_per_layer

        print(f"   {num_threads:2d}   |  {ms_per_layer:.2f}   | {tok_s:.2f} | {speedup:.2f}x")

        results.append({
            "threads": num_threads,
            "ms_per_layer": ms_per_layer,
            "tok_s": tok_s,
            "speedup_vs_1t": speedup,
        })

    # Find optimal
    best = max(results, key=lambda r: r["tok_s"])
    print(f"\nOptimal: {best['threads']} threads @ {best['tok_s']:.2f} tok/s")

    output = {
        "cpu_model": cpu_model,
        "results": results,
        "optimal_threads": best["threads"],
        "optimal_tok_s": best["tok_s"],
        "timestamp": datetime.now().isoformat(),
    }

    # Save result
    results_file = Path("/results/thread_sweep.jsonl")
    with open(results_file, "a") as f:
        f.write(json.dumps(output) + "\n")
    results_volume.commit()

    return output


@app.function(
    image=modal.Image.debian_slim(),
    volumes={"/results": results_volume},
)
def get_results(limit: int = 20) -> list[dict[str, Any]]:
    """Get recent benchmark results."""
    results_file = Path("/results/bench_history.jsonl")
    if not results_file.exists():
        return []

    results = []
    with open(results_file) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    return results[-limit:]


@app.local_entrypoint()
def main(
    iterations: int = 1,
    threads: int = 8,
    sweep: bool = False,
    list_results: bool = False,
):
    """CLI entry point for Modal CPU benchmark.

    Examples:
        # Run single benchmark
        modal run benchmark/modal_cpu_bench.py

        # Run 20 iterations
        modal run benchmark/modal_cpu_bench.py --iterations 20

        # Thread sweep
        modal run benchmark/modal_cpu_bench.py --sweep

        # List results
        modal run benchmark/modal_cpu_bench.py --list-results
    """
    if list_results:
        results = get_results.remote()
        for r in results:
            print(json.dumps(r, indent=2))
        return

    if sweep:
        print("Running thread sweep...")
        result = thread_sweep.remote()
        print(json.dumps(result, indent=2))
        return

    print(f"Running {iterations} benchmark iteration(s)...")
    for i in range(iterations):
        print(f"\n=== Iteration {i + 1}/{iterations} ===")
        result = benchmark_native.remote(num_threads=threads)
        print(f"Result: {result['native']['tok_s']:.2f} tok/s")
