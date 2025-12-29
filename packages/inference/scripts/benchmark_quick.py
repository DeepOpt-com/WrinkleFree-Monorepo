#!/usr/bin/env python3
"""Quick benchmark for sglang-bitnet performance.

Measures:
- Tokens/second (prefill + decode)
- Time-to-first-token (TTFT)
- Latency per output token
- Throughput at different sequence lengths

Usage:
    python scripts/benchmark_quick.py --output benchmark_results/iter_1.json
    python scripts/benchmark_quick.py --url http://192.168.1.217:30000
"""

import argparse
import json
import os
import statistics
import time
import concurrent.futures
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

import requests

SGLANG_URL = os.environ.get("SGLANG_URL", "http://127.0.0.1:30000")


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    prompt_tokens: int
    completion_tokens: int
    total_time_s: float
    ttft_s: float  # time to first token
    tokens_per_second: float
    latency_per_token_ms: float


@dataclass
class BenchmarkSummary:
    """Summary statistics for a benchmark configuration."""
    config: str
    prompt_tokens: int
    max_tokens: int
    runs: int
    tokens_per_second_mean: float
    tokens_per_second_std: float
    ttft_mean_ms: float
    ttft_p95_ms: float
    latency_per_token_mean_ms: float
    total_time_mean_s: float


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

    # Estimate prompt tokens (rough approximation)
    prompt_tokens = len(prompt.split()) * 1.3  # rough estimate

    return BenchmarkResult(
        prompt_tokens=int(prompt_tokens),
        completion_tokens=token_count,
        total_time_s=total_time,
        ttft_s=ttft,
        tokens_per_second=token_count / total_time if total_time > 0 else 0,
        latency_per_token_ms=(total_time * 1000 / token_count) if token_count > 0 else 0,
    )


def generate_sync(
    url: str,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.0,
) -> BenchmarkResult:
    """Generate synchronously for simpler benchmarking."""
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    start_time = time.perf_counter()
    resp = requests.post(
        f"{url}/v1/chat/completions",
        json=payload,
        timeout=300,
    )
    end_time = time.perf_counter()

    data = resp.json()
    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_time = end_time - start_time

    return BenchmarkResult(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_time_s=total_time,
        ttft_s=total_time,  # Can't measure TTFT without streaming
        tokens_per_second=completion_tokens / total_time if total_time > 0 else 0,
        latency_per_token_ms=(total_time * 1000 / completion_tokens) if completion_tokens > 0 else 0,
    )


def make_prompt(target_tokens: int) -> str:
    """Create a prompt of approximately target_tokens length."""
    base = "Tell me an interesting fact. "
    # Rough estimate: 1 word ~ 1.3 tokens
    words_needed = int(target_tokens / 1.3)
    return base * (words_needed // len(base.split()) + 1)


def run_benchmark(
    url: str,
    prompt_tokens: int,
    max_tokens: int,
    runs: int = 5,
    use_streaming: bool = True,
) -> BenchmarkSummary:
    """Run benchmark with given configuration."""
    prompt = make_prompt(prompt_tokens)
    results: list[BenchmarkResult] = []

    generate_fn = generate_streaming if use_streaming else generate_sync

    print(f"  Benchmarking prompt={prompt_tokens}tok, max={max_tokens}tok, {runs} runs...")

    for i in range(runs):
        try:
            result = generate_fn(url, prompt, max_tokens)
            results.append(result)
            print(f"    Run {i+1}: {result.tokens_per_second:.1f} tok/s, TTFT={result.ttft_s*1000:.0f}ms")
        except Exception as e:
            print(f"    Run {i+1}: ERROR - {e}")

    if not results:
        return BenchmarkSummary(
            config=f"prompt={prompt_tokens},max={max_tokens}",
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            runs=0,
            tokens_per_second_mean=0,
            tokens_per_second_std=0,
            ttft_mean_ms=0,
            ttft_p95_ms=0,
            latency_per_token_mean_ms=0,
            total_time_mean_s=0,
        )

    tps_values = [r.tokens_per_second for r in results]
    ttft_values = [r.ttft_s * 1000 for r in results]  # Convert to ms
    latency_values = [r.latency_per_token_ms for r in results]
    time_values = [r.total_time_s for r in results]

    return BenchmarkSummary(
        config=f"prompt={prompt_tokens},max={max_tokens}",
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        runs=len(results),
        tokens_per_second_mean=statistics.mean(tps_values),
        tokens_per_second_std=statistics.stdev(tps_values) if len(tps_values) > 1 else 0,
        ttft_mean_ms=statistics.mean(ttft_values),
        ttft_p95_ms=sorted(ttft_values)[int(len(ttft_values) * 0.95)] if len(ttft_values) > 1 else ttft_values[0],
        latency_per_token_mean_ms=statistics.mean(latency_values),
        total_time_mean_s=statistics.mean(time_values),
    )


def check_server(url: str) -> dict:
    """Check server status and get model info."""
    try:
        resp = requests.get(f"{url}/v1/models", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            models = data.get("data", [])
            if models:
                return {"status": "ok", "model": models[0].get("id", "unknown")}
        return {"status": "error", "message": "No models loaded"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def run_concurrent_benchmark(
    url: str,
    num_requests: int,
    max_tokens: int = 50,
    prompt: str = "Hello, how are you?",
) -> dict:
    """Run concurrent requests to test batch parallelism and continuous batching."""

    def single_request(request_id: int) -> dict:
        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": f"{prompt} (Request {request_id})"}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": False,
        }

        start = time.perf_counter()
        try:
            resp = requests.post(
                f"{url}/v1/chat/completions",
                json=payload,
                timeout=120,
            )
            elapsed = time.perf_counter() - start
            data = resp.json()
            usage = data.get("usage", {})
            return {
                "success": True,
                "request_id": request_id,
                "time_s": elapsed,
                "completion_tokens": usage.get("completion_tokens", 0),
            }
        except Exception as e:
            return {
                "success": False,
                "request_id": request_id,
                "time_s": time.perf_counter() - start,
                "error": str(e),
            }

    print(f"  Running {num_requests} concurrent requests...")
    start_time = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = [executor.submit(single_request, i) for i in range(num_requests)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    total_time = time.perf_counter() - start_time

    # Calculate metrics
    successful = [r for r in results if r["success"]]
    total_tokens = sum(r["completion_tokens"] for r in successful)
    avg_latency = statistics.mean([r["time_s"] for r in successful]) if successful else 0

    return {
        "num_requests": num_requests,
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "total_time_s": total_time,
        "total_tokens": total_tokens,
        "throughput_req_per_s": len(successful) / total_time if total_time > 0 else 0,
        "throughput_tok_per_s": total_tokens / total_time if total_time > 0 else 0,
        "avg_latency_s": avg_latency,
    }


def main():
    parser = argparse.ArgumentParser(description="Quick benchmark for sglang-bitnet")
    parser.add_argument("--url", default=SGLANG_URL, help="SGLang server URL")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    parser.add_argument("--runs", type=int, default=5, help="Runs per configuration")
    parser.add_argument("--streaming", action="store_true", default=True, help="Use streaming")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer configs)")
    args = parser.parse_args()

    # Check server
    print(f"Connecting to {args.url}...")
    server_info = check_server(args.url)
    if server_info["status"] != "ok":
        print(f"Error: {server_info['message']}")
        return 1

    print(f"Server OK - Model: {server_info['model']}")

    # Define benchmark configurations
    if args.quick:
        configs = [
            (32, 50),   # Short prompt, short output
            (128, 100), # Medium
        ]
    else:
        configs = [
            (32, 50),    # Short prompt, short output
            (32, 100),   # Short prompt, medium output
            (32, 200),   # Short prompt, long output
            (128, 50),   # Medium prompt, short output
            (128, 100),  # Medium prompt, medium output
            (128, 200),  # Medium prompt, long output
            (512, 50),   # Long prompt, short output
            (512, 100),  # Long prompt, medium output
        ]

    # Run benchmarks
    print(f"\n{'='*60}")
    print(f"Running benchmarks ({args.runs} runs each)...")
    print(f"{'='*60}\n")

    results = []
    for prompt_tokens, max_tokens in configs:
        summary = run_benchmark(
            args.url,
            prompt_tokens,
            max_tokens,
            runs=args.runs,
            use_streaming=args.streaming,
        )
        results.append(summary)
        print()

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<25} {'Tok/s':>10} {'TTFT(ms)':>10} {'Latency(ms)':>12}")
    print("-" * 60)

    for r in results:
        print(f"{r.config:<25} {r.tokens_per_second_mean:>10.1f} {r.ttft_mean_ms:>10.0f} {r.latency_per_token_mean_ms:>12.1f}")

    # Calculate overall averages
    if results:
        avg_tps = statistics.mean([r.tokens_per_second_mean for r in results])
        avg_ttft = statistics.mean([r.ttft_mean_ms for r in results])
        avg_latency = statistics.mean([r.latency_per_token_mean_ms for r in results])
        print("-" * 60)
        print(f"{'AVERAGE':<25} {avg_tps:>10.1f} {avg_ttft:>10.0f} {avg_latency:>12.1f}")

    # Concurrent batch testing
    batch_results = []
    if not args.quick:
        print(f"\n{'='*60}")
        print("CONCURRENT BATCH TESTING (Continuous Batching)")
        print(f"{'='*60}")

        for num_concurrent in [2, 4, 8]:
            result = run_concurrent_benchmark(args.url, num_concurrent, max_tokens=50)
            batch_results.append(result)
            print(f"    {num_concurrent} concurrent: {result['throughput_tok_per_s']:.1f} tok/s total, "
                  f"{result['throughput_req_per_s']:.2f} req/s, "
                  f"{result['avg_latency_s']*1000:.0f}ms avg latency")

    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "server_url": args.url,
            "model": server_info["model"],
            "runs_per_config": args.runs,
            "streaming": args.streaming,
            "results": [asdict(r) for r in results],
            "averages": {
                "tokens_per_second": avg_tps if results else 0,
                "ttft_ms": avg_ttft if results else 0,
                "latency_per_token_ms": avg_latency if results else 0,
            },
        }
        if not args.quick and batch_results:
            output_data["batch_results"] = batch_results
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
