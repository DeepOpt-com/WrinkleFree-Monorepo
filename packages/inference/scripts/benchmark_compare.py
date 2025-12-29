#!/usr/bin/env python3
"""Benchmark comparison: sglang-bitnet vs BitNet.cpp

Sequentially benchmarks both inference backends and compares performance.

Usage:
    python scripts/benchmark_compare.py
    python scripts/benchmark_compare.py --output results.json
    python scripts/benchmark_compare.py --sglang-only
    python scripts/benchmark_compare.py --bitnet-only
"""

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

# Project root (WrinkleFree-Inference-Engine)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Default model
DEFAULT_MODEL = "microsoft/bitnet-b1.58-2B-4T"
DEFAULT_GGUF_MODEL = PROJECT_ROOT / "extern/BitNet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"

# Backend configurations
BACKENDS = {
    "sglang-bitnet": {
        "port": 30000,
        "health_endpoint": "/v1/models",
    },
    "bitnet-cpp": {
        "port": 8080,
        "health_endpoint": "/",  # llama.cpp returns 200 OK on root
    },
}


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    prompt_tokens: int
    completion_tokens: int
    total_time_s: float
    ttft_s: float
    tokens_per_second: float
    latency_per_token_ms: float


@dataclass
class BackendResults:
    """Aggregated results for a backend."""
    name: str
    tokens_per_second_mean: float
    tokens_per_second_std: float
    ttft_mean_ms: float
    latency_per_token_mean_ms: float
    total_runs: int
    configs_tested: list


def check_prerequisites(run_sglang: bool, run_bitnet: bool) -> list[str]:
    """Check prerequisites and return list of errors."""
    errors = []

    if run_sglang:
        # Check sglang-bitnet
        venv_python = PROJECT_ROOT / ".venv/bin/python"
        if not venv_python.exists():
            errors.append(f"Missing .venv - run: cd {PROJECT_ROOT} && uv sync")

        # Check sglang is installed
        try:
            result = subprocess.run(
                [str(venv_python), "-c", "import sglang; print('ok')"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                errors.append("sglang not installed - see CLAUDE.md for setup instructions")
        except Exception as e:
            errors.append(f"Cannot check sglang installation: {e}")

    if run_bitnet:
        # Check bitnet.cpp binary
        bitnet_server = PROJECT_ROOT / "extern/BitNet/build/bin/llama-server"
        if not bitnet_server.exists():
            errors.append(
                "BitNet.cpp not built - run:\n"
                f"  cd {PROJECT_ROOT}/extern/BitNet\n"
                "  python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s"
            )

        # Check GGUF model
        if not DEFAULT_GGUF_MODEL.exists():
            errors.append(
                f"GGUF model missing at {DEFAULT_GGUF_MODEL}\n"
                "  Run setup_env.py or download from HuggingFace"
            )

    return errors


def start_sglang_server(model: str, threads: int, port: int) -> subprocess.Popen:
    """Start sglang-bitnet server."""
    venv_python = PROJECT_ROOT / ".venv/bin/python"
    cmd = [
        str(venv_python),
        "-m", "sglang.launch_server",
        "--model-path", model,
        "--port", str(port),
        "--device", "cpu",
        "--tp", "1",
    ]
    print(f"Starting sglang-bitnet: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
    )
    return proc


def start_bitnet_server(model: str, threads: int, port: int) -> subprocess.Popen:
    """Start BitNet.cpp server."""
    server_path = PROJECT_ROOT / "extern/BitNet/build/bin/llama-server"
    cmd = [
        str(server_path),
        "-m", str(model),
        "-c", "2048",
        "-t", str(threads),
        "--host", "127.0.0.1",
        "--port", str(port),
        "-cb",  # continuous batching
    ]
    print(f"Starting BitNet.cpp: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(PROJECT_ROOT / "extern/BitNet"),
    )
    return proc


def wait_for_ready(url: str, health_endpoint: str, timeout: int = 300) -> bool:
    """Wait for server to be ready."""
    start_time = time.time()
    check_url = f"{url}{health_endpoint}"

    print(f"Waiting for server at {check_url}...", end="", flush=True)
    while time.time() - start_time < timeout:
        try:
            resp = requests.get(check_url, timeout=5)
            if resp.status_code == 200:
                print(" ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        print(".", end="", flush=True)
        time.sleep(2)

    print(" timeout!")
    return False


def stop_backend(proc: subprocess.Popen, timeout: int = 10):
    """Stop backend server gracefully."""
    if proc.poll() is None:
        print("Stopping server...")
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            print("Force killing server...")
            proc.kill()
            proc.wait()


def generate_streaming(
    url: str,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.0,
) -> BenchmarkResult:
    """Generate with streaming to measure TTFT."""
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    start_time = time.perf_counter()
    first_token_time = None
    token_count = 0

    with requests.post(
        f"{url}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=300,
    ) as resp:
        resp.raise_for_status()

        for line in resp.iter_lines():
            if not line:
                continue

            line_str = line.decode("utf-8")
            if not line_str.startswith("data: "):
                continue

            data_str = line_str[6:]
            if data_str == "[DONE]":
                break

            try:
                data = json.loads(data_str)
                choices = data.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        token_count += 1
            except json.JSONDecodeError:
                continue

    end_time = time.perf_counter()
    total_time = end_time - start_time
    ttft = (first_token_time - start_time) if first_token_time else total_time

    # Estimate prompt tokens
    prompt_tokens = int(len(prompt.split()) * 1.3)

    return BenchmarkResult(
        prompt_tokens=prompt_tokens,
        completion_tokens=token_count,
        total_time_s=total_time,
        ttft_s=ttft,
        tokens_per_second=token_count / total_time if total_time > 0 else 0,
        latency_per_token_ms=(total_time * 1000 / token_count) if token_count > 0 else 0,
    )


def make_prompt(target_tokens: int) -> str:
    """Create a prompt of approximately target_tokens length."""
    base = "Tell me an interesting fact. "
    words_needed = int(target_tokens / 1.3)
    return base * (words_needed // len(base.split()) + 1)


def run_benchmark_config(
    url: str,
    prompt_tokens: int,
    max_tokens: int,
    runs: int = 3,
) -> list[BenchmarkResult]:
    """Run benchmark with given configuration."""
    prompt = make_prompt(prompt_tokens)
    results = []

    print(f"  Config: prompt={prompt_tokens}tok, max={max_tokens}tok, {runs} runs")

    for i in range(runs):
        try:
            result = generate_streaming(url, prompt, max_tokens)
            results.append(result)
            print(f"    Run {i+1}: {result.tokens_per_second:.1f} tok/s, TTFT={result.ttft_s*1000:.0f}ms")
        except Exception as e:
            print(f"    Run {i+1}: ERROR - {e}")

    return results


def benchmark_backend(
    name: str,
    url: str,
    runs: int = 3,
    quick: bool = False,
    warmup_runs: int = 2,
) -> BackendResults:
    """Run full benchmark suite for a backend."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"{'='*60}")

    # Warmup runs (not timed) to JIT compile and warm caches
    if warmup_runs > 0:
        print(f"  Warmup: {warmup_runs} runs to warm JIT cache...")
        for i in range(warmup_runs):
            try:
                _ = generate_streaming(url, "Hello, how are you?", max_tokens=5)
                print(f"    Warmup {i+1}: OK")
            except Exception as e:
                print(f"    Warmup {i+1}: ERROR - {e}")
        print()

    # Benchmark configurations
    if quick:
        configs = [
            (32, 50),
            (128, 100),
        ]
    else:
        configs = [
            (32, 50),
            (32, 100),
            (128, 50),
            (128, 100),
        ]

    all_results = []
    configs_tested = []

    for prompt_tokens, max_tokens in configs:
        results = run_benchmark_config(url, prompt_tokens, max_tokens, runs=runs)
        all_results.extend(results)
        configs_tested.append(f"prompt={prompt_tokens},max={max_tokens}")

    if not all_results:
        return BackendResults(
            name=name,
            tokens_per_second_mean=0,
            tokens_per_second_std=0,
            ttft_mean_ms=0,
            latency_per_token_mean_ms=0,
            total_runs=0,
            configs_tested=configs_tested,
        )

    tps_values = [r.tokens_per_second for r in all_results]
    ttft_values = [r.ttft_s * 1000 for r in all_results]
    latency_values = [r.latency_per_token_ms for r in all_results]

    return BackendResults(
        name=name,
        tokens_per_second_mean=statistics.mean(tps_values),
        tokens_per_second_std=statistics.stdev(tps_values) if len(tps_values) > 1 else 0,
        ttft_mean_ms=statistics.mean(ttft_values),
        latency_per_token_mean_ms=statistics.mean(latency_values),
        total_runs=len(all_results),
        configs_tested=configs_tested,
    )


def print_comparison(sglang_results: Optional[BackendResults], bitnet_results: Optional[BackendResults]):
    """Print comparison table."""
    print(f"\n{'='*70}")
    print("BENCHMARK COMPARISON: sglang-bitnet vs BitNet.cpp")
    print(f"{'='*70}")

    if sglang_results and bitnet_results:
        # Calculate speedups (sglang relative to bitnet)
        tps_speedup = sglang_results.tokens_per_second_mean / bitnet_results.tokens_per_second_mean if bitnet_results.tokens_per_second_mean > 0 else 0
        ttft_speedup = bitnet_results.ttft_mean_ms / sglang_results.ttft_mean_ms if sglang_results.ttft_mean_ms > 0 else 0
        latency_speedup = bitnet_results.latency_per_token_mean_ms / sglang_results.latency_per_token_mean_ms if sglang_results.latency_per_token_mean_ms > 0 else 0

        print(f"\n{'Metric':<25} {'sglang-bitnet':>15} {'bitnet.cpp':>15} {'Speedup':>10}")
        print("-" * 70)
        print(f"{'Tokens/sec (decode)':<25} {sglang_results.tokens_per_second_mean:>15.1f} {bitnet_results.tokens_per_second_mean:>15.1f} {tps_speedup:>9.2f}x")
        print(f"{'Time to First Token':<25} {sglang_results.ttft_mean_ms:>14.0f}ms {bitnet_results.ttft_mean_ms:>14.0f}ms {ttft_speedup:>9.2f}x")
        print(f"{'Latency per Token':<25} {sglang_results.latency_per_token_mean_ms:>13.1f}ms {bitnet_results.latency_per_token_mean_ms:>13.1f}ms {latency_speedup:>9.2f}x")
        print("-" * 70)
        print(f"{'Total Runs':<25} {sglang_results.total_runs:>15} {bitnet_results.total_runs:>15}")
    elif sglang_results:
        print(f"\nOnly sglang-bitnet results available:")
        print(f"  Tokens/sec: {sglang_results.tokens_per_second_mean:.1f}")
        print(f"  TTFT: {sglang_results.ttft_mean_ms:.0f}ms")
        print(f"  Latency/token: {sglang_results.latency_per_token_mean_ms:.1f}ms")
    elif bitnet_results:
        print(f"\nOnly BitNet.cpp results available:")
        print(f"  Tokens/sec: {bitnet_results.tokens_per_second_mean:.1f}")
        print(f"  TTFT: {bitnet_results.ttft_mean_ms:.0f}ms")
        print(f"  Latency/token: {bitnet_results.latency_per_token_mean_ms:.1f}ms")

    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark comparison: sglang-bitnet vs BitNet.cpp")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model for sglang")
    parser.add_argument("--gguf-model", type=Path, default=DEFAULT_GGUF_MODEL, help="GGUF model for BitNet.cpp")
    parser.add_argument("--threads", type=int, default=os.cpu_count(), help="Thread count")
    parser.add_argument("--runs", type=int, default=3, help="Runs per configuration")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    parser.add_argument("--sglang-only", action="store_true", help="Run only sglang-bitnet")
    parser.add_argument("--bitnet-only", action="store_true", help="Run only BitNet.cpp")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer configs)")
    args = parser.parse_args()

    run_sglang = not args.bitnet_only
    run_bitnet = not args.sglang_only

    print(f"Benchmark Comparison")
    print(f"  Threads: {args.threads}")
    print(f"  Runs per config: {args.runs}")
    print(f"  Backends: {', '.join(b for b, run in [('sglang-bitnet', run_sglang), ('bitnet.cpp', run_bitnet)] if run)}")

    # Check prerequisites
    errors = check_prerequisites(run_sglang, run_bitnet)
    if errors:
        print("\nPrerequisites missing:")
        for e in errors:
            print(f"  - {e}")
        return 1

    sglang_results = None
    bitnet_results = None
    proc = None

    try:
        # Benchmark sglang-bitnet
        if run_sglang:
            port = BACKENDS["sglang-bitnet"]["port"]
            proc = start_sglang_server(args.model, args.threads, port)

            url = f"http://127.0.0.1:{port}"
            if not wait_for_ready(url, BACKENDS["sglang-bitnet"]["health_endpoint"]):
                print("ERROR: sglang-bitnet failed to start")
                stop_backend(proc)
                return 1

            sglang_results = benchmark_backend("sglang-bitnet", url, runs=args.runs, quick=args.quick)
            stop_backend(proc)
            proc = None

            # Wait a bit for port to be released
            time.sleep(2)

        # Benchmark BitNet.cpp
        if run_bitnet:
            port = BACKENDS["bitnet-cpp"]["port"]
            proc = start_bitnet_server(args.gguf_model, args.threads, port)

            url = f"http://127.0.0.1:{port}"
            if not wait_for_ready(url, BACKENDS["bitnet-cpp"]["health_endpoint"]):
                print("ERROR: BitNet.cpp failed to start")
                stop_backend(proc)
                return 1

            bitnet_results = benchmark_backend("bitnet.cpp", url, runs=args.runs, quick=args.quick)
            stop_backend(proc)
            proc = None

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        if proc:
            stop_backend(proc)
        return 1
    finally:
        if proc:
            stop_backend(proc)

    # Print comparison
    print_comparison(sglang_results, bitnet_results)

    # Save results
    if args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "threads": args.threads,
            "runs_per_config": args.runs,
            "quick_mode": args.quick,
        }
        if sglang_results:
            output_data["sglang_bitnet"] = asdict(sglang_results)
        if bitnet_results:
            output_data["bitnet_cpp"] = asdict(bitnet_results)
        if sglang_results and bitnet_results:
            output_data["comparison"] = {
                "tps_speedup": sglang_results.tokens_per_second_mean / bitnet_results.tokens_per_second_mean if bitnet_results.tokens_per_second_mean > 0 else 0,
                "ttft_speedup": bitnet_results.ttft_mean_ms / sglang_results.ttft_mean_ms if sglang_results.ttft_mean_ms > 0 else 0,
                "latency_speedup": bitnet_results.latency_per_token_mean_ms / sglang_results.latency_per_token_mean_ms if sglang_results.latency_per_token_mean_ms > 0 else 0,
            }

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
