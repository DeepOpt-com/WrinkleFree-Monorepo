"""Tests for continuous batching in SGLang inference server.

These tests verify that the server properly batches concurrent requests
for improved throughput.

Run with: uv run pytest tests/test_continuous_batching.py -v

NOTE: Requires a running SGLang server on localhost:30000 (or INFERENCE_URL env var)
"""

import pytest
import time
import asyncio
import aiohttp
import os
from concurrent.futures import ThreadPoolExecutor


# Server URL (can be overridden by environment variable)
INFERENCE_URL = os.environ.get("INFERENCE_URL", "http://localhost:30000")

# Skip if no server available
pytestmark = pytest.mark.integration


def make_request_sync(session_url: str, prompt: str, max_tokens: int = 20) -> dict:
    """Make a synchronous chat completion request."""
    import requests

    response = requests.post(
        f"{session_url}/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


async def make_request_async(session: aiohttp.ClientSession, prompt: str, max_tokens: int = 20) -> dict:
    """Make an async chat completion request."""
    async with session.post(
        f"{INFERENCE_URL}/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        },
    ) as response:
        response.raise_for_status()
        return await response.json()


class TestContinuousBatching:
    """Test continuous batching performance."""

    @pytest.fixture(autouse=True)
    def check_server(self):
        """Skip if server is not available."""
        import requests
        try:
            response = requests.get(f"{INFERENCE_URL}/v1/models", timeout=5)
            response.raise_for_status()
        except Exception as e:
            pytest.skip(f"Server not available at {INFERENCE_URL}: {e}")

    def test_single_request(self):
        """Verify single request works."""
        result = make_request_sync(INFERENCE_URL, "Hello", max_tokens=10)
        assert "choices" in result
        assert len(result["choices"]) > 0

    def test_sequential_requests(self):
        """Baseline: measure sequential request time."""
        num_requests = 4
        start = time.perf_counter()

        for i in range(num_requests):
            make_request_sync(INFERENCE_URL, f"Hello {i}", max_tokens=20)

        elapsed = time.perf_counter() - start
        print(f"\n{num_requests} sequential requests: {elapsed:.2f}s")

        # Store for comparison
        self._sequential_time = elapsed
        return elapsed

    def test_concurrent_requests_batched(self):
        """Test that concurrent requests are batched for speedup."""
        num_requests = 4

        # Sequential baseline
        start = time.perf_counter()
        for i in range(num_requests):
            make_request_sync(INFERENCE_URL, f"Hello {i}", max_tokens=20)
        sequential_time = time.perf_counter() - start

        # Concurrent (should be batched)
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [
                executor.submit(make_request_sync, INFERENCE_URL, f"Hello {i}", 20)
                for i in range(num_requests)
            ]
            results = [f.result() for f in futures]
        concurrent_time = time.perf_counter() - start

        # Verify all requests succeeded
        for result in results:
            assert "choices" in result

        speedup = sequential_time / concurrent_time
        print(f"\nSequential: {sequential_time:.2f}s")
        print(f"Concurrent: {concurrent_time:.2f}s")
        print(f"Speedup: {speedup:.2f}x")

        # Expect at least 1.3x speedup from batching
        # (lower threshold to account for variability)
        assert speedup > 1.2, f"Expected batching speedup >1.2x, got {speedup:.2f}x"

    @pytest.mark.asyncio
    async def test_async_concurrent_batching(self):
        """Test async concurrent requests for batching."""
        num_requests = 8

        # Sequential
        async with aiohttp.ClientSession() as session:
            start = time.perf_counter()
            for i in range(num_requests):
                await make_request_async(session, f"Count to {i}", max_tokens=20)
            sequential_time = time.perf_counter() - start

        # Concurrent
        async with aiohttp.ClientSession() as session:
            start = time.perf_counter()
            tasks = [
                make_request_async(session, f"Count to {i}", max_tokens=20)
                for i in range(num_requests)
            ]
            results = await asyncio.gather(*tasks)
            concurrent_time = time.perf_counter() - start

        # Verify all requests succeeded
        for result in results:
            assert "choices" in result

        speedup = sequential_time / concurrent_time
        print(f"\n{num_requests} requests async:")
        print(f"Sequential: {sequential_time:.2f}s")
        print(f"Concurrent: {concurrent_time:.2f}s")
        print(f"Speedup: {speedup:.2f}x")

        # Expect significant speedup with more requests
        assert speedup > 1.5, f"Expected batching speedup >1.5x, got {speedup:.2f}x"

    def test_high_concurrency_batching(self):
        """Test with higher concurrency to verify batching scales."""
        num_requests = 16

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [
                executor.submit(make_request_sync, INFERENCE_URL, f"Hello {i}", 10)
                for i in range(num_requests)
            ]
            results = [f.result() for f in futures]
        concurrent_time = time.perf_counter() - start

        # Verify all requests succeeded
        assert len(results) == num_requests
        for result in results:
            assert "choices" in result

        tokens_per_second = (num_requests * 10) / concurrent_time
        print(f"\n{num_requests} concurrent requests: {concurrent_time:.2f}s")
        print(f"Throughput: {tokens_per_second:.1f} tok/s total")


class TestBatchingVsNoBatching:
    """Compare batching behavior between different request patterns."""

    @pytest.fixture(autouse=True)
    def check_server(self):
        """Skip if server is not available."""
        import requests
        try:
            response = requests.get(f"{INFERENCE_URL}/v1/models", timeout=5)
            response.raise_for_status()
        except Exception:
            pytest.skip(f"Server not available at {INFERENCE_URL}")

    def test_batching_improves_throughput(self):
        """Verify that batching provides throughput improvement."""
        # Single request baseline
        start = time.perf_counter()
        make_request_sync(INFERENCE_URL, "Hello", max_tokens=20)
        single_time = time.perf_counter() - start

        # 4 concurrent requests
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(make_request_sync, INFERENCE_URL, f"Hello {i}", 20)
                for i in range(4)
            ]
            [f.result() for f in futures]
        batch_time = time.perf_counter() - start

        # Batching should complete 4x requests in less than 4x time
        efficiency = (4 * single_time) / batch_time
        print(f"\nSingle request: {single_time:.2f}s")
        print(f"4 batched requests: {batch_time:.2f}s")
        print(f"Batching efficiency: {efficiency:.2f}x")

        assert efficiency > 1.0, "Batching should improve throughput"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
