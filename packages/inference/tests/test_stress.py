"""Stress tests for inference engine under concurrent load."""

import asyncio
import statistics
import time
from dataclasses import dataclass, field

import pytest

from wrinklefree_inference.client.bitnet_client import AsyncBitNetClient, BitNetClient


@dataclass
class StressTestResults:
    """Results from a stress test run."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_time_seconds: float = 0.0
    requests_per_second: float = 0.0
    latencies_ms: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests

    @property
    def avg_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return statistics.mean(self.latencies_ms)

    @property
    def p50_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return statistics.median(self.latencies_ms)

    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_latencies = sorted(self.latencies_ms)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    @property
    def p99_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_latencies = sorted(self.latencies_ms)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    def summary(self) -> str:
        return (
            f"Stress Test Results:\n"
            f"  Total requests:     {self.total_requests}\n"
            f"  Successful:         {self.successful_requests}\n"
            f"  Failed:             {self.failed_requests}\n"
            f"  Success rate:       {self.success_rate*100:.1f}%\n"
            f"  Total time:         {self.total_time_seconds:.2f}s\n"
            f"  Requests/second:    {self.requests_per_second:.2f}\n"
            f"  Avg latency:        {self.avg_latency_ms:.1f}ms\n"
            f"  P50 latency:        {self.p50_latency_ms:.1f}ms\n"
            f"  P95 latency:        {self.p95_latency_ms:.1f}ms\n"
            f"  P99 latency:        {self.p99_latency_ms:.1f}ms\n"
        )


async def run_stress_test(
    base_url: str,
    num_requests: int = 100,
    concurrency: int = 10,
    max_tokens: int = 20,
    timeout: int = 60,
) -> StressTestResults:
    """
    Run a stress test with concurrent requests.

    Args:
        base_url: Server URL
        num_requests: Total number of requests to send
        concurrency: Number of concurrent requests
        max_tokens: Tokens per request
        timeout: Request timeout

    Returns:
        StressTestResults with metrics
    """
    results = StressTestResults()
    results.total_requests = num_requests

    prompts = [
        f"Question {i}: Explain concept number {i} briefly."
        for i in range(num_requests)
    ]

    semaphore = asyncio.Semaphore(concurrency)

    async def make_request(client: AsyncBitNetClient, prompt: str) -> tuple[bool, float, str]:
        async with semaphore:
            start = time.perf_counter()
            try:
                await client.generate(prompt, max_tokens=max_tokens)
                latency = (time.perf_counter() - start) * 1000
                return True, latency, ""
            except Exception as e:
                latency = (time.perf_counter() - start) * 1000
                return False, latency, str(e)

    start_time = time.perf_counter()

    async with AsyncBitNetClient(base_url, timeout) as client:
        tasks = [make_request(client, prompt) for prompt in prompts]
        responses = await asyncio.gather(*tasks)

    results.total_time_seconds = time.perf_counter() - start_time

    for success, latency, error in responses:
        if success:
            results.successful_requests += 1
            results.latencies_ms.append(latency)
        else:
            results.failed_requests += 1
            if error and error not in results.errors:
                results.errors.append(error)

    if results.total_time_seconds > 0:
        results.requests_per_second = results.total_requests / results.total_time_seconds

    return results


class TestStressLight:
    """Light stress tests (fewer requests, faster execution)."""

    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_concurrent_10(self, inference_url: str, skip_if_no_server):
        """Test 10 concurrent requests."""
        results = await run_stress_test(
            inference_url,
            num_requests=10,
            concurrency=10,
            max_tokens=10,
        )
        print(f"\n{results.summary()}")

        assert results.success_rate >= 0.8, f"Success rate too low: {results.success_rate}"

    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_concurrent_25(self, inference_url: str, skip_if_no_server):
        """Test 25 concurrent requests."""
        results = await run_stress_test(
            inference_url,
            num_requests=25,
            concurrency=25,
            max_tokens=10,
        )
        print(f"\n{results.summary()}")

        assert results.success_rate >= 0.8, f"Success rate too low: {results.success_rate}"

    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_sequential_burst(self, inference_url: str, skip_if_no_server):
        """Test sequential requests in rapid succession."""
        results = await run_stress_test(
            inference_url,
            num_requests=20,
            concurrency=1,  # Sequential
            max_tokens=10,
        )
        print(f"\n{results.summary()}")

        assert results.success_rate >= 0.95, f"Sequential should be reliable"


class TestStressMedium:
    """Medium stress tests."""

    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_concurrent_50(self, inference_url: str, skip_if_no_server):
        """Test 50 concurrent requests."""
        results = await run_stress_test(
            inference_url,
            num_requests=50,
            concurrency=50,
            max_tokens=15,
        )
        print(f"\n{results.summary()}")

        assert results.success_rate >= 0.7, f"Success rate too low: {results.success_rate}"

    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_sustained_load(self, inference_url: str, skip_if_no_server):
        """Test sustained load over multiple batches."""
        all_results = []

        for batch in range(3):
            results = await run_stress_test(
                inference_url,
                num_requests=20,
                concurrency=10,
                max_tokens=10,
            )
            all_results.append(results)
            print(f"Batch {batch + 1}: {results.success_rate*100:.0f}% success, "
                  f"{results.requests_per_second:.1f} req/s")

        # All batches should maintain good performance
        avg_success = sum(r.success_rate for r in all_results) / len(all_results)
        assert avg_success >= 0.8, f"Average success rate too low: {avg_success}"


class TestStressHeavy:
    """Heavy stress tests (use with caution)."""

    @pytest.mark.stress
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_concurrent_100(self, inference_url: str, skip_if_no_server):
        """Test 100 concurrent requests."""
        results = await run_stress_test(
            inference_url,
            num_requests=100,
            concurrency=100,
            max_tokens=10,
            timeout=120,
        )
        print(f"\n{results.summary()}")

        # Heavy load may have lower success rate
        assert results.success_rate >= 0.5, f"Success rate too low: {results.success_rate}"

    @pytest.mark.stress
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_mixed_load_pattern(self, inference_url: str, skip_if_no_server):
        """Test mixed load pattern (varying concurrency)."""
        patterns = [
            (10, 5),   # Light
            (30, 30),  # Burst
            (10, 2),   # Recovery
            (50, 25),  # Heavy
            (10, 5),   # Light again
        ]

        all_results = []
        for num_requests, concurrency in patterns:
            results = await run_stress_test(
                inference_url,
                num_requests=num_requests,
                concurrency=concurrency,
                max_tokens=10,
            )
            all_results.append(results)
            print(f"Pattern ({num_requests}, {concurrency}): "
                  f"{results.success_rate*100:.0f}% success")

        # Overall should maintain reasonable success
        total_success = sum(r.successful_requests for r in all_results)
        total_requests = sum(r.total_requests for r in all_results)
        overall_rate = total_success / total_requests
        assert overall_rate >= 0.7, f"Overall success rate too low: {overall_rate}"


class TestStressRecovery:
    """Tests for server recovery after stress."""

    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_recovery_after_burst(self, inference_url: str, skip_if_no_server):
        """Test that server recovers after burst load."""
        # Send burst
        burst_results = await run_stress_test(
            inference_url,
            num_requests=30,
            concurrency=30,
            max_tokens=10,
        )
        print(f"Burst: {burst_results.success_rate*100:.0f}% success")

        # Wait briefly
        await asyncio.sleep(2)

        # Check recovery with single request
        async with AsyncBitNetClient(inference_url) as client:
            start = time.perf_counter()
            response = await client.generate("Hello", max_tokens=5)
            latency = (time.perf_counter() - start) * 1000

        print(f"Recovery request: {latency:.0f}ms")
        assert response is not None, "Server should recover after burst"

    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_error_handling_under_load(self, inference_url: str, skip_if_no_server):
        """Test that errors are handled gracefully under load."""
        # Mix of valid and potentially problematic requests
        async with AsyncBitNetClient(inference_url, timeout=30) as client:
            tasks = []

            # Normal requests
            for i in range(10):
                tasks.append(client.generate(f"Normal question {i}", max_tokens=10))

            # Very short prompts
            for i in range(5):
                tasks.append(client.generate("", max_tokens=5))

            # Longer prompts
            for i in range(5):
                tasks.append(client.generate("word " * 100, max_tokens=10))

            results = await asyncio.gather(*tasks, return_exceptions=True)

        successes = sum(1 for r in results if not isinstance(r, Exception))
        print(f"Mixed load: {successes}/{len(results)} succeeded")

        # Most should succeed
        assert successes >= 15, f"Too many failures: {len(results) - successes}"
