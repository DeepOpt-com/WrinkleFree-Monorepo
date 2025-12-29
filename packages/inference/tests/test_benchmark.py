"""Benchmark tests for measuring inference performance."""

import asyncio
import statistics
import time
from dataclasses import dataclass, field
from typing import Optional

import pytest

from wrinklefree_inference.client.bitnet_client import AsyncBitNetClient, BitNetClient


@dataclass
class BenchmarkResults:
    """Results from a benchmark run."""

    name: str
    iterations: int = 0
    total_tokens_generated: int = 0
    total_time_seconds: float = 0.0

    # Latency metrics (in milliseconds)
    time_to_first_token_ms: list[float] = field(default_factory=list)
    total_latencies_ms: list[float] = field(default_factory=list)

    # Throughput metrics
    tokens_per_second: float = 0.0
    requests_per_second: float = 0.0

    @property
    def avg_ttft_ms(self) -> float:
        """Average time to first token."""
        if not self.time_to_first_token_ms:
            return 0.0
        return statistics.mean(self.time_to_first_token_ms)

    @property
    def p50_ttft_ms(self) -> float:
        if not self.time_to_first_token_ms:
            return 0.0
        return statistics.median(self.time_to_first_token_ms)

    @property
    def p95_ttft_ms(self) -> float:
        if not self.time_to_first_token_ms:
            return 0.0
        sorted_ttft = sorted(self.time_to_first_token_ms)
        idx = int(len(sorted_ttft) * 0.95)
        return sorted_ttft[min(idx, len(sorted_ttft) - 1)]

    @property
    def avg_latency_ms(self) -> float:
        if not self.total_latencies_ms:
            return 0.0
        return statistics.mean(self.total_latencies_ms)

    @property
    def p50_latency_ms(self) -> float:
        if not self.total_latencies_ms:
            return 0.0
        return statistics.median(self.total_latencies_ms)

    @property
    def p95_latency_ms(self) -> float:
        if not self.total_latencies_ms:
            return 0.0
        sorted_lat = sorted(self.total_latencies_ms)
        idx = int(len(sorted_lat) * 0.95)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    def summary(self) -> str:
        return (
            f"Benchmark: {self.name}\n"
            f"{'=' * 50}\n"
            f"  Iterations:           {self.iterations}\n"
            f"  Total tokens:         {self.total_tokens_generated}\n"
            f"  Total time:           {self.total_time_seconds:.2f}s\n"
            f"\n"
            f"  Throughput:\n"
            f"    Tokens/second:      {self.tokens_per_second:.2f}\n"
            f"    Requests/second:    {self.requests_per_second:.2f}\n"
            f"\n"
            f"  Time to First Token:\n"
            f"    Average:            {self.avg_ttft_ms:.1f}ms\n"
            f"    P50:                {self.p50_ttft_ms:.1f}ms\n"
            f"    P95:                {self.p95_ttft_ms:.1f}ms\n"
            f"\n"
            f"  Total Latency:\n"
            f"    Average:            {self.avg_latency_ms:.1f}ms\n"
            f"    P50:                {self.p50_latency_ms:.1f}ms\n"
            f"    P95:                {self.p95_latency_ms:.1f}ms\n"
        )


async def benchmark_streaming(
    client: AsyncBitNetClient,
    prompt: str,
    max_tokens: int,
) -> tuple[float, float, int]:
    """
    Benchmark a single streaming request.

    Returns:
        (time_to_first_token_ms, total_latency_ms, tokens_generated)
    """
    tokens_generated = 0
    first_token_time: Optional[float] = None

    start = time.perf_counter()

    async for token in client.generate_stream(prompt, max_tokens=max_tokens):
        if first_token_time is None:
            first_token_time = time.perf_counter()
        tokens_generated += 1

    end = time.perf_counter()

    ttft_ms = ((first_token_time or end) - start) * 1000
    total_ms = (end - start) * 1000

    return ttft_ms, total_ms, tokens_generated


async def benchmark_batch(
    client: AsyncBitNetClient,
    prompt: str,
    max_tokens: int,
) -> tuple[float, int]:
    """
    Benchmark a single non-streaming request.

    Returns:
        (total_latency_ms, estimated_tokens)
    """
    start = time.perf_counter()
    response = await client.generate(prompt, max_tokens=max_tokens)
    end = time.perf_counter()

    # Estimate tokens from response length
    estimated_tokens = len(response.split())

    return (end - start) * 1000, estimated_tokens


class TestBenchmarkTTFT:
    """Time to First Token benchmarks."""

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_ttft_short_prompt(self, inference_url: str, skip_if_no_server):
        """Benchmark TTFT with short prompt."""
        results = BenchmarkResults(name="TTFT - Short Prompt")

        async with AsyncBitNetClient(inference_url) as client:
            for i in range(10):
                ttft, total, tokens = await benchmark_streaming(
                    client, "Hello", max_tokens=20
                )
                results.time_to_first_token_ms.append(ttft)
                results.total_latencies_ms.append(total)
                results.total_tokens_generated += tokens
                results.iterations += 1

        results.total_time_seconds = sum(results.total_latencies_ms) / 1000
        results.tokens_per_second = (
            results.total_tokens_generated / results.total_time_seconds
            if results.total_time_seconds > 0 else 0
        )
        results.requests_per_second = (
            results.iterations / results.total_time_seconds
            if results.total_time_seconds > 0 else 0
        )

        print(f"\n{results.summary()}")

        # Basic performance expectations
        assert results.avg_ttft_ms < 5000, f"TTFT too slow: {results.avg_ttft_ms}ms"

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_ttft_medium_prompt(self, inference_url: str, skip_if_no_server):
        """Benchmark TTFT with medium-length prompt."""
        results = BenchmarkResults(name="TTFT - Medium Prompt")
        prompt = "Explain the concept of machine learning in simple terms. " * 5

        async with AsyncBitNetClient(inference_url) as client:
            for i in range(10):
                ttft, total, tokens = await benchmark_streaming(
                    client, prompt, max_tokens=30
                )
                results.time_to_first_token_ms.append(ttft)
                results.total_latencies_ms.append(total)
                results.total_tokens_generated += tokens
                results.iterations += 1

        results.total_time_seconds = sum(results.total_latencies_ms) / 1000
        results.tokens_per_second = (
            results.total_tokens_generated / results.total_time_seconds
            if results.total_time_seconds > 0 else 0
        )

        print(f"\n{results.summary()}")

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_ttft_long_prompt(self, inference_url: str, skip_if_no_server):
        """Benchmark TTFT with long prompt (context-heavy)."""
        results = BenchmarkResults(name="TTFT - Long Prompt")
        prompt = "word " * 500 + "What is the meaning of the above?"

        async with AsyncBitNetClient(inference_url) as client:
            for i in range(5):
                ttft, total, tokens = await benchmark_streaming(
                    client, prompt, max_tokens=20
                )
                results.time_to_first_token_ms.append(ttft)
                results.total_latencies_ms.append(total)
                results.total_tokens_generated += tokens
                results.iterations += 1

        results.total_time_seconds = sum(results.total_latencies_ms) / 1000

        print(f"\n{results.summary()}")


class TestBenchmarkThroughput:
    """Throughput benchmarks."""

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_throughput_sequential(self, inference_url: str, skip_if_no_server):
        """Benchmark sequential request throughput."""
        results = BenchmarkResults(name="Throughput - Sequential")
        prompt = "Count to ten: 1, 2, 3,"

        start_total = time.perf_counter()

        async with AsyncBitNetClient(inference_url) as client:
            for i in range(20):
                latency, tokens = await benchmark_batch(client, prompt, max_tokens=30)
                results.total_latencies_ms.append(latency)
                results.total_tokens_generated += tokens
                results.iterations += 1

        results.total_time_seconds = time.perf_counter() - start_total
        results.tokens_per_second = (
            results.total_tokens_generated / results.total_time_seconds
        )
        results.requests_per_second = results.iterations / results.total_time_seconds

        print(f"\n{results.summary()}")

        # Minimum throughput expectation
        assert results.tokens_per_second > 1, "Throughput too low"

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_throughput_concurrent(self, inference_url: str, skip_if_no_server):
        """Benchmark concurrent request throughput."""
        results = BenchmarkResults(name="Throughput - Concurrent (10)")
        prompt = "What is 2+2?"
        concurrency = 10

        async def single_request(client, prompt):
            return await benchmark_batch(client, prompt, max_tokens=20)

        start_total = time.perf_counter()

        async with AsyncBitNetClient(inference_url, timeout=60) as client:
            # Run 5 batches of concurrent requests
            for batch in range(5):
                tasks = [single_request(client, prompt) for _ in range(concurrency)]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in batch_results:
                    if not isinstance(result, Exception):
                        latency, tokens = result
                        results.total_latencies_ms.append(latency)
                        results.total_tokens_generated += tokens
                        results.iterations += 1

        results.total_time_seconds = time.perf_counter() - start_total
        results.tokens_per_second = (
            results.total_tokens_generated / results.total_time_seconds
        )
        results.requests_per_second = results.iterations / results.total_time_seconds

        print(f"\n{results.summary()}")

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_throughput_varying_lengths(self, inference_url: str, skip_if_no_server):
        """Benchmark throughput with varying output lengths."""
        results = BenchmarkResults(name="Throughput - Varying Lengths")
        prompt = "Tell me a story:"

        token_counts = [10, 20, 50, 100, 50, 20, 10]  # Varying lengths

        start_total = time.perf_counter()

        async with AsyncBitNetClient(inference_url, timeout=120) as client:
            for max_tokens in token_counts:
                latency, tokens = await benchmark_batch(client, prompt, max_tokens)
                results.total_latencies_ms.append(latency)
                results.total_tokens_generated += tokens
                results.iterations += 1

        results.total_time_seconds = time.perf_counter() - start_total
        results.tokens_per_second = (
            results.total_tokens_generated / results.total_time_seconds
        )

        print(f"\n{results.summary()}")


class TestBenchmarkLatency:
    """Latency distribution benchmarks."""

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_latency_distribution(self, inference_url: str, skip_if_no_server):
        """Benchmark latency distribution over many requests."""
        results = BenchmarkResults(name="Latency Distribution")
        prompt = "Quick question: What is AI?"

        async with AsyncBitNetClient(inference_url) as client:
            for i in range(50):
                latency, tokens = await benchmark_batch(client, prompt, max_tokens=20)
                results.total_latencies_ms.append(latency)
                results.total_tokens_generated += tokens
                results.iterations += 1

        results.total_time_seconds = sum(results.total_latencies_ms) / 1000

        print(f"\n{results.summary()}")

        # Check latency variance
        if len(results.total_latencies_ms) > 1:
            std_dev = statistics.stdev(results.total_latencies_ms)
            print(f"  Latency Std Dev:      {std_dev:.1f}ms")

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_cold_vs_warm_latency(self, inference_url: str, skip_if_no_server):
        """Compare cold start vs warm cache latency."""
        prompt_a = "Unique prompt A for cold start test"
        prompt_b = "Different prompt B for comparison"

        async with AsyncBitNetClient(inference_url) as client:
            # Cold start (first request)
            start = time.perf_counter()
            await client.generate(prompt_a, max_tokens=20)
            cold_latency = (time.perf_counter() - start) * 1000

            # Warm (same prompt)
            start = time.perf_counter()
            await client.generate(prompt_a, max_tokens=20)
            warm_latency = (time.perf_counter() - start) * 1000

            # Different prompt (may still benefit from warm cache)
            start = time.perf_counter()
            await client.generate(prompt_b, max_tokens=20)
            different_latency = (time.perf_counter() - start) * 1000

        print(f"\nCold vs Warm Latency:")
        print(f"  Cold (first request):     {cold_latency:.1f}ms")
        print(f"  Warm (same prompt):       {warm_latency:.1f}ms")
        print(f"  Different prompt:         {different_latency:.1f}ms")

        if cold_latency > 0:
            print(f"  Warm speedup:             {cold_latency/warm_latency:.2f}x")


class TestBenchmarkScaling:
    """Scaling benchmarks."""

    @pytest.mark.benchmark
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_scaling_with_concurrency(self, inference_url: str, skip_if_no_server):
        """Benchmark how throughput scales with concurrency."""
        prompt = "Hello"
        concurrency_levels = [1, 2, 5, 10, 20]

        print("\nScaling with Concurrency:")
        print("-" * 50)

        for concurrency in concurrency_levels:
            async def single_request(client):
                start = time.perf_counter()
                await client.generate(prompt, max_tokens=10)
                return time.perf_counter() - start

            start_total = time.perf_counter()

            async with AsyncBitNetClient(inference_url, timeout=60) as client:
                tasks = [single_request(client) for _ in range(concurrency * 2)]
                results = await asyncio.gather(*tasks, return_exceptions=True)

            total_time = time.perf_counter() - start_total
            successful = sum(1 for r in results if not isinstance(r, Exception))
            rps = successful / total_time if total_time > 0 else 0

            print(f"  Concurrency {concurrency:2d}: {rps:.1f} req/s "
                  f"({successful}/{len(results)} success)")

    @pytest.mark.benchmark
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_scaling_with_output_length(self, inference_url: str, skip_if_no_server):
        """Benchmark how latency scales with output length."""
        prompt = "Continue this story:"
        output_lengths = [10, 25, 50, 100, 200]

        print("\nScaling with Output Length:")
        print("-" * 50)

        async with AsyncBitNetClient(inference_url, timeout=120) as client:
            for max_tokens in output_lengths:
                latencies = []
                for _ in range(3):
                    start = time.perf_counter()
                    response = await client.generate(prompt, max_tokens=max_tokens)
                    latency = (time.perf_counter() - start) * 1000
                    latencies.append(latency)

                avg_latency = statistics.mean(latencies)
                tokens_per_sec = max_tokens / (avg_latency / 1000) if avg_latency > 0 else 0

                print(f"  {max_tokens:3d} tokens: {avg_latency:.0f}ms avg "
                      f"({tokens_per_sec:.1f} tok/s)")
