#!/usr/bin/env python3
"""
Benchmark throughput for inference server.

Measures tokens/second at different batch sizes to help calculate
cost per million tokens.

Usage:
    # Against local server
    uv run python scripts/benchmark_throughput.py --endpoint http://localhost:8080

    # Against deployed service
    uv run python scripts/benchmark_throughput.py --endpoint https://my-service.sky.serve

    # Custom parameters
    uv run python scripts/benchmark_throughput.py \
        --endpoint http://localhost:8080 \
        --batch-sizes 1,4,8,16 \
        --duration 30 \
        --max-tokens 100
"""

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass
from typing import Optional

try:
    import aiohttp
except ImportError:
    print("Please install aiohttp: pip install aiohttp")
    exit(1)


@dataclass
class BenchmarkResult:
    batch_size: int
    total_requests: int
    total_tokens: int
    duration_seconds: float
    tokens_per_second: float
    requests_per_second: float
    latency_p50_ms: float
    latency_p99_ms: float
    errors: int


# Sample prompts of varying lengths
PROMPTS = [
    "What is machine learning?",
    "Explain the concept of neural networks in simple terms.",
    "Write a short poem about the ocean.",
    "What are the benefits of cloud computing for businesses?",
    "Describe the process of photosynthesis step by step.",
    "What is the difference between AI and machine learning?",
    "How does encryption work to protect data?",
    "Explain quantum computing to a beginner.",
]


async def make_request(
    session: aiohttp.ClientSession,
    endpoint: str,
    prompt: str,
    max_tokens: int,
) -> tuple[int, float, Optional[str]]:
    """Make a single inference request. Returns (tokens_generated, latency_ms, error)."""
    start = time.perf_counter()

    try:
        async with session.post(
            f"{endpoint}/v1/completions",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.7,
            },
            timeout=aiohttp.ClientTimeout(total=120),
        ) as response:
            latency = (time.perf_counter() - start) * 1000

            if response.status != 200:
                return 0, latency, f"HTTP {response.status}"

            data = await response.json()

            # Extract token count from response
            if "usage" in data:
                tokens = data["usage"].get("completion_tokens", max_tokens)
            elif "choices" in data and len(data["choices"]) > 0:
                # Estimate from text length if no usage info
                text = data["choices"][0].get("text", "")
                tokens = len(text.split()) * 1.3  # Rough estimate
            else:
                tokens = max_tokens

            return int(tokens), latency, None

    except asyncio.TimeoutError:
        latency = (time.perf_counter() - start) * 1000
        return 0, latency, "Timeout"
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return 0, latency, str(e)


async def run_batch(
    session: aiohttp.ClientSession,
    endpoint: str,
    batch_size: int,
    max_tokens: int,
) -> list[tuple[int, float, Optional[str]]]:
    """Run a batch of concurrent requests."""
    tasks = []
    for i in range(batch_size):
        prompt = PROMPTS[i % len(PROMPTS)]
        tasks.append(make_request(session, endpoint, prompt, max_tokens))

    return await asyncio.gather(*tasks)


async def benchmark_batch_size(
    endpoint: str,
    batch_size: int,
    duration_seconds: int,
    max_tokens: int,
) -> BenchmarkResult:
    """Benchmark a specific batch size for the given duration."""

    all_latencies: list[float] = []
    total_tokens = 0
    total_requests = 0
    errors = 0

    start_time = time.perf_counter()

    async with aiohttp.ClientSession() as session:
        while time.perf_counter() - start_time < duration_seconds:
            results = await run_batch(session, endpoint, batch_size, max_tokens)

            for tokens, latency, error in results:
                total_requests += 1
                if error:
                    errors += 1
                else:
                    total_tokens += tokens
                    all_latencies.append(latency)

    actual_duration = time.perf_counter() - start_time

    # Calculate statistics
    if all_latencies:
        sorted_latencies = sorted(all_latencies)
        p50_idx = int(len(sorted_latencies) * 0.50)
        p99_idx = int(len(sorted_latencies) * 0.99)
        latency_p50 = sorted_latencies[p50_idx]
        latency_p99 = sorted_latencies[min(p99_idx, len(sorted_latencies) - 1)]
    else:
        latency_p50 = 0
        latency_p99 = 0

    return BenchmarkResult(
        batch_size=batch_size,
        total_requests=total_requests,
        total_tokens=total_tokens,
        duration_seconds=actual_duration,
        tokens_per_second=total_tokens / actual_duration if actual_duration > 0 else 0,
        requests_per_second=total_requests / actual_duration if actual_duration > 0 else 0,
        latency_p50_ms=latency_p50,
        latency_p99_ms=latency_p99,
        errors=errors,
    )


async def check_health(endpoint: str) -> bool:
    """Check if the endpoint is healthy."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{endpoint}/health",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                return response.status == 200
    except Exception:
        return False


def calculate_costs(tokens_per_second: float) -> dict:
    """Calculate cost per million tokens for different infrastructures."""
    tokens_per_hour = tokens_per_second * 3600

    if tokens_per_hour == 0:
        return {}

    infrastructures = {
        "Hetzner AX42 ($0.075/hr)": 0.075,
        "Hetzner AX102 ($0.214/hr)": 0.214,
        "OVHCloud b3-64 ($0.16/hr)": 0.16,
        "AWS r7a.4xl spot ($0.20/hr)": 0.20,
    }

    costs = {}
    for name, hourly_cost in infrastructures.items():
        cost_per_million = (hourly_cost / tokens_per_hour) * 1_000_000
        costs[name] = {
            "100% util": round(cost_per_million, 4),
            "70% util": round(cost_per_million / 0.70, 4),
            "50% util": round(cost_per_million / 0.50, 4),
        }

    return costs


async def main():
    parser = argparse.ArgumentParser(
        description="Benchmark inference throughput",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8080",
        help="Inference server endpoint (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--batch-sizes",
        default="1,4,8,16,32",
        help="Comma-separated batch sizes to test (default: 1,4,8,16,32)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Duration in seconds for each batch size (default: 30)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Max tokens per request (default: 50)",
    )
    parser.add_argument(
        "--output",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )

    args = parser.parse_args()

    endpoint = args.endpoint.rstrip("/")
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]

    # Check health
    print(f"Checking endpoint: {endpoint}")
    if not await check_health(endpoint):
        print(f"ERROR: Endpoint {endpoint} is not healthy")
        print("Make sure the inference server is running.")
        return 1
    print("Endpoint is healthy.\n")

    # Run benchmarks
    results: list[BenchmarkResult] = []

    for batch_size in batch_sizes:
        print(f"Benchmarking batch_size={batch_size} for {args.duration}s...")
        result = await benchmark_batch_size(
            endpoint=endpoint,
            batch_size=batch_size,
            duration_seconds=args.duration,
            max_tokens=args.max_tokens,
        )
        results.append(result)
        print(f"  -> {result.tokens_per_second:.1f} tok/s, {result.errors} errors")

    print("\n" + "=" * 80)

    # Output results
    if args.output == "json":
        output = {
            "endpoint": endpoint,
            "max_tokens": args.max_tokens,
            "duration_per_test": args.duration,
            "results": [
                {
                    "batch_size": r.batch_size,
                    "tokens_per_second": round(r.tokens_per_second, 2),
                    "requests_per_second": round(r.requests_per_second, 2),
                    "latency_p50_ms": round(r.latency_p50_ms, 1),
                    "latency_p99_ms": round(r.latency_p99_ms, 1),
                    "total_requests": r.total_requests,
                    "errors": r.errors,
                }
                for r in results
            ],
        }

        # Add cost analysis for best throughput
        best = max(results, key=lambda r: r.tokens_per_second)
        output["best_throughput"] = {
            "batch_size": best.batch_size,
            "tokens_per_second": round(best.tokens_per_second, 2),
            "cost_per_million_tokens": calculate_costs(best.tokens_per_second),
        }

        print(json.dumps(output, indent=2))
    else:
        # Table output
        print("\nBENCHMARK RESULTS")
        print("=" * 80)
        print(f"{'Batch Size':>10} | {'Tok/s':>10} | {'Req/s':>8} | {'P50 (ms)':>10} | {'P99 (ms)':>10} | {'Errors':>6}")
        print("-" * 80)

        for r in results:
            print(f"{r.batch_size:>10} | {r.tokens_per_second:>10.1f} | {r.requests_per_second:>8.1f} | {r.latency_p50_ms:>10.1f} | {r.latency_p99_ms:>10.1f} | {r.errors:>6}")

        print("\n" + "=" * 80)

        # Find optimal batch size (best throughput)
        best = max(results, key=lambda r: r.tokens_per_second)
        print(f"\nOPTIMAL BATCH SIZE: {best.batch_size}")
        print(f"  Throughput: {best.tokens_per_second:.1f} tokens/second")
        print(f"  Latency P99: {best.latency_p99_ms:.1f}ms")

        # Cost analysis
        print("\n" + "=" * 80)
        print("\nCOST PER MILLION TOKENS (at optimal batch size)")
        print("-" * 80)

        costs = calculate_costs(best.tokens_per_second)
        for infra, utilization_costs in costs.items():
            print(f"\n{infra}:")
            for util, cost in utilization_costs.items():
                print(f"  {util}: ${cost:.4f}/1M tokens")

        print("\n" + "=" * 80)
        print("\nSUGGESTED PRICING (to achieve target margins)")
        print("-" * 80)

        # Use Hetzner AX102 as reference
        base_cost = costs.get("Hetzner AX102 ($0.214/hr)", {}).get("70% util", 0.30)
        print(f"\nBase cost (Hetzner @ 70% util): ${base_cost:.4f}/1M tokens")
        print(f"\n{'Margin':>10} | {'Price':>15} | {'Monthly Revenue @ 1B tokens':>30}")
        print("-" * 60)
        for margin in [0.3, 0.5, 0.7, 0.8]:
            price = base_cost / (1 - margin)
            monthly = price * 1000  # 1B tokens = 1000 × 1M
            print(f"{margin*100:>9.0f}% | ${price:>14.4f} | ${monthly:>29,.0f}")

    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
