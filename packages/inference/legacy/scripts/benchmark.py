#!/usr/bin/env python3
"""Script to run comprehensive benchmarks against inference server."""

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wrinklefree_inference.client.bitnet_client import AsyncBitNetClient


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""

    url: str = "http://localhost:8080"
    warmup_requests: int = 5
    benchmark_requests: int = 50
    max_tokens: int = 50
    concurrency_levels: list[int] = field(default_factory=lambda: [1, 5, 10, 20])
    output_format: str = "table"  # table, json


@dataclass
class BenchmarkMetrics:
    """Metrics from a benchmark run."""

    name: str
    requests: int
    successful: int
    failed: int
    total_time_seconds: float
    tokens_generated: int

    # Latency percentiles (ms)
    latency_avg: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    latency_min: float
    latency_max: float

    # Throughput
    requests_per_second: float
    tokens_per_second: float

    @classmethod
    def from_latencies(
        cls,
        name: str,
        latencies: list[float],
        tokens: list[int],
        total_time: float,
        failed: int = 0,
    ) -> "BenchmarkMetrics":
        if not latencies:
            return cls(
                name=name, requests=0, successful=0, failed=failed,
                total_time_seconds=total_time, tokens_generated=0,
                latency_avg=0, latency_p50=0, latency_p95=0, latency_p99=0,
                latency_min=0, latency_max=0,
                requests_per_second=0, tokens_per_second=0,
            )

        sorted_lat = sorted(latencies)
        total_tokens = sum(tokens)

        return cls(
            name=name,
            requests=len(latencies) + failed,
            successful=len(latencies),
            failed=failed,
            total_time_seconds=total_time,
            tokens_generated=total_tokens,
            latency_avg=statistics.mean(latencies),
            latency_p50=statistics.median(latencies),
            latency_p95=sorted_lat[int(len(sorted_lat) * 0.95)] if len(sorted_lat) > 1 else sorted_lat[0],
            latency_p99=sorted_lat[int(len(sorted_lat) * 0.99)] if len(sorted_lat) > 1 else sorted_lat[0],
            latency_min=min(latencies),
            latency_max=max(latencies),
            requests_per_second=len(latencies) / total_time if total_time > 0 else 0,
            tokens_per_second=total_tokens / total_time if total_time > 0 else 0,
        )


async def run_benchmark(
    client: AsyncBitNetClient,
    name: str,
    prompts: list[str],
    max_tokens: int,
    concurrency: int = 1,
) -> BenchmarkMetrics:
    """Run a benchmark with given configuration."""
    latencies: list[float] = []
    tokens: list[int] = []
    failed = 0

    semaphore = asyncio.Semaphore(concurrency)

    async def single_request(prompt: str) -> tuple[Optional[float], int]:
        async with semaphore:
            try:
                start = time.perf_counter()
                response = await client.generate(prompt, max_tokens=max_tokens)
                latency = (time.perf_counter() - start) * 1000
                # Estimate tokens from words
                token_count = len(response.split())
                return latency, token_count
            except Exception:
                return None, 0

    start_total = time.perf_counter()
    tasks = [single_request(prompt) for prompt in prompts]
    results = await asyncio.gather(*tasks)
    total_time = time.perf_counter() - start_total

    for latency, token_count in results:
        if latency is not None:
            latencies.append(latency)
            tokens.append(token_count)
        else:
            failed += 1

    return BenchmarkMetrics.from_latencies(name, latencies, tokens, total_time, failed)


def print_table(metrics_list: list[BenchmarkMetrics]):
    """Print metrics as formatted table."""
    print("\n" + "=" * 100)
    print(f"{'Benchmark':<30} {'Req/s':>8} {'Tok/s':>8} {'Avg':>8} {'P50':>8} {'P95':>8} {'P99':>8} {'Success':>10}")
    print("=" * 100)

    for m in metrics_list:
        success_rate = f"{m.successful}/{m.requests}"
        print(
            f"{m.name:<30} "
            f"{m.requests_per_second:>8.2f} "
            f"{m.tokens_per_second:>8.1f} "
            f"{m.latency_avg:>7.0f}ms "
            f"{m.latency_p50:>7.0f}ms "
            f"{m.latency_p95:>7.0f}ms "
            f"{m.latency_p99:>7.0f}ms "
            f"{success_rate:>10}"
        )

    print("=" * 100)


def print_json(metrics_list: list[BenchmarkMetrics]):
    """Print metrics as JSON."""
    data = [asdict(m) for m in metrics_list]
    print(json.dumps(data, indent=2))


async def main():
    parser = argparse.ArgumentParser(description="Run inference benchmarks")
    parser.add_argument(
        "--url",
        default="http://localhost:8080",
        help="Inference server URL",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup requests",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=50,
        help="Number of benchmark requests per test",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Max tokens per request",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="Concurrency levels to test",
    )
    parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format",
    )

    args = parser.parse_args()

    print(f"Connecting to {args.url}...")

    async with AsyncBitNetClient(args.url, timeout=120) as client:
        # Health check
        if not await client.health_check():
            print(f"Error: Server not available at {args.url}")
            sys.exit(1)

        print("Server is healthy. Starting benchmarks...\n")

        # Warmup
        print(f"Warming up with {args.warmup} requests...")
        for i in range(args.warmup):
            await client.generate("Warmup request", max_tokens=10)
        print("Warmup complete.\n")

        metrics_list: list[BenchmarkMetrics] = []

        # Test 1: Short prompts
        prompts = ["Hello, how are you?"] * args.requests
        result = await run_benchmark(
            client, "Short Prompt (Sequential)",
            prompts, args.max_tokens, concurrency=1
        )
        metrics_list.append(result)
        print(f"  {result.name}: {result.requests_per_second:.2f} req/s")

        # Test 2: Medium prompts
        prompts = ["Explain the concept of machine learning briefly."] * args.requests
        result = await run_benchmark(
            client, "Medium Prompt (Sequential)",
            prompts, args.max_tokens, concurrency=1
        )
        metrics_list.append(result)
        print(f"  {result.name}: {result.requests_per_second:.2f} req/s")

        # Test 3: Varying concurrency levels
        prompts = ["What is AI?"] * args.requests
        for concurrency in args.concurrency:
            result = await run_benchmark(
                client, f"Concurrent ({concurrency})",
                prompts, args.max_tokens, concurrency=concurrency
            )
            metrics_list.append(result)
            print(f"  {result.name}: {result.requests_per_second:.2f} req/s")

        # Test 4: Long output
        prompts = ["Tell me a detailed story:"] * (args.requests // 2)
        result = await run_benchmark(
            client, "Long Output (100 tokens)",
            prompts, max_tokens=100, concurrency=1
        )
        metrics_list.append(result)
        print(f"  {result.name}: {result.tokens_per_second:.1f} tok/s")

        # Output results
        if args.format == "table":
            print_table(metrics_list)
        else:
            print_json(metrics_list)


if __name__ == "__main__":
    asyncio.run(main())
