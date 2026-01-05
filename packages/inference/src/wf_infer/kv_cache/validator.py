"""KV cache validation utilities for BitNet inference.

This module provides tools to validate that KV caching is working correctly:
- Prefix caching: Same prefix should result in faster second request
- Context limits: Graceful handling at context window boundaries
- Continuous batching: Concurrent requests should not fail
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from wf_infer.client.bitnet_client import AsyncBitNetClient, BitNetClient

logger = logging.getLogger(__name__)


@dataclass
class KVCacheMetrics:
    """Metrics from KV cache validation."""

    prefix_speedup: float = 0.0
    """Ratio of first request latency to second request latency (>1 = cache working)"""

    context_limit_handled: bool = False
    """Whether the context limit was handled gracefully"""

    concurrent_success_rate: float = 0.0
    """Fraction of concurrent requests that succeeded"""

    first_request_latency_ms: float = 0.0
    """Latency of first request in milliseconds"""

    second_request_latency_ms: float = 0.0
    """Latency of second request (same prefix) in milliseconds"""

    errors: list[str] = field(default_factory=list)
    """Any errors encountered during validation"""


class KVCacheValidator:
    """
    Validate KV cache behavior during inference.

    Tests:
    1. Prefix caching - repeated prompts with same prefix should be faster
    2. Context limits - graceful handling at context window boundary
    3. Continuous batching - concurrent requests should succeed

    Args:
        base_url: BitNet server URL
        timeout: Request timeout in seconds
    """

    def __init__(self, base_url: str = "http://localhost:8080", timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.sync_client = BitNetClient(
            host=base_url.replace("http://", "").split(":")[0],
            port=int(base_url.split(":")[-1]) if ":" in base_url.split("/")[-1] else 8080,
            timeout=timeout,
        )
        # Override base_url directly
        self.sync_client.base_url = self.base_url

    async def validate_all(self) -> KVCacheMetrics:
        """
        Run all KV cache validation tests.

        Returns:
            KVCacheMetrics with results from all tests
        """
        metrics = KVCacheMetrics()

        # Test 1: Prefix caching
        try:
            prefix_result = await self.validate_prefix_caching()
            metrics.prefix_speedup = prefix_result["speedup"]
            metrics.first_request_latency_ms = prefix_result["first_latency_ms"]
            metrics.second_request_latency_ms = prefix_result["second_latency_ms"]
        except Exception as e:
            metrics.errors.append(f"Prefix caching test failed: {e}")
            logger.error(f"Prefix caching test failed: {e}")

        # Test 2: Context limits
        try:
            metrics.context_limit_handled = await self.validate_context_limits()
        except Exception as e:
            metrics.errors.append(f"Context limit test failed: {e}")
            logger.error(f"Context limit test failed: {e}")

        # Test 3: Continuous batching
        try:
            metrics.concurrent_success_rate = await self.validate_continuous_batching()
        except Exception as e:
            metrics.errors.append(f"Continuous batching test failed: {e}")
            logger.error(f"Continuous batching test failed: {e}")

        return metrics

    async def validate_prefix_caching(
        self,
        prefix: str = "The following is a detailed explanation of machine learning:\n\n",
        suffix1: str = "What is supervised learning?",
        suffix2: str = "What is unsupervised learning?",
    ) -> dict:
        """
        Verify that common prefixes are cached efficiently.

        Strategy:
        1. Send prefix + suffix_1
        2. Measure latency
        3. Send prefix + suffix_2 (same prefix)
        4. Second latency should be lower (cache hit on prefix)

        Returns:
            Dict with speedup ratio and latencies
        """
        async with AsyncBitNetClient(self.base_url, self.timeout) as client:
            # First request with prefix
            start = time.perf_counter()
            await client.generate(prefix + suffix1, max_tokens=50)
            first_latency = time.perf_counter() - start

            # Small delay to ensure server is ready
            await asyncio.sleep(0.1)

            # Second request with same prefix
            start = time.perf_counter()
            await client.generate(prefix + suffix2, max_tokens=50)
            second_latency = time.perf_counter() - start

        speedup = first_latency / second_latency if second_latency > 0 else 1.0

        result = {
            "first_latency_ms": first_latency * 1000,
            "second_latency_ms": second_latency * 1000,
            "speedup": speedup,
        }

        logger.info(
            f"Prefix caching: first={result['first_latency_ms']:.1f}ms, "
            f"second={result['second_latency_ms']:.1f}ms, speedup={speedup:.2f}x"
        )

        return result

    async def validate_context_limits(
        self,
        context_size: int = 4096,
        tokens_per_word: float = 1.3,
    ) -> bool:
        """
        Verify behavior at context window boundaries.

        Strategy:
        1. Generate prompt that approaches context limit
        2. Verify response is generated without error
        3. Generate prompt that exceeds context
        4. Verify graceful handling (truncation or error, not crash)

        Returns:
            True if context limits are handled gracefully
        """
        async with AsyncBitNetClient(self.base_url, self.timeout) as client:
            # Generate a prompt that's ~80% of context
            target_tokens = int(context_size * 0.8)
            words_needed = int(target_tokens / tokens_per_word)
            long_prompt = "test " * words_needed

            # Should succeed
            try:
                response = await client.generate(long_prompt, max_tokens=10)
                logger.info(f"Near-limit prompt succeeded, got {len(response)} chars")
            except Exception as e:
                logger.warning(f"Near-limit prompt failed: {e}")
                return False

            # Generate prompt that exceeds context
            exceed_prompt = "word " * (context_size + 100)

            try:
                # This might truncate or error, both are acceptable
                response = await client.generate(exceed_prompt, max_tokens=10)
                logger.info(f"Exceed-limit prompt handled, got {len(response)} chars")
                return True
            except Exception as e:
                # An error is acceptable if it's graceful
                if "context" in str(e).lower() or "too long" in str(e).lower():
                    logger.info(f"Exceed-limit prompt gracefully rejected: {e}")
                    return True
                logger.warning(f"Exceed-limit prompt failed unexpectedly: {e}")
                return False

    async def validate_continuous_batching(
        self,
        num_concurrent: int = 10,
        max_tokens: int = 20,
    ) -> float:
        """
        Verify continuous batching works under load.

        Strategy:
        1. Send multiple concurrent requests
        2. Verify all complete successfully
        3. Return success rate

        Returns:
            Fraction of successful requests (0.0 to 1.0)
        """
        prompts = [
            f"Question {i}: What is the capital of country number {i}?"
            for i in range(num_concurrent)
        ]

        async with AsyncBitNetClient(self.base_url, self.timeout) as client:
            tasks = [
                client.generate(prompt, max_tokens=max_tokens)
                for prompt in prompts
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

        successes = sum(1 for r in results if not isinstance(r, Exception))
        success_rate = successes / len(results)

        errors = [str(r) for r in results if isinstance(r, Exception)]
        if errors:
            logger.warning(f"Concurrent request errors: {errors[:3]}")

        logger.info(
            f"Continuous batching: {successes}/{len(results)} succeeded "
            f"({success_rate*100:.1f}%)"
        )

        return success_rate


def run_kv_cache_validation(
    base_url: str = "http://localhost:8080",
    timeout: int = 60,
) -> KVCacheMetrics:
    """
    Run KV cache validation synchronously.

    Args:
        base_url: BitNet server URL
        timeout: Request timeout

    Returns:
        KVCacheMetrics with validation results
    """
    validator = KVCacheValidator(base_url, timeout)
    return asyncio.run(validator.validate_all())
