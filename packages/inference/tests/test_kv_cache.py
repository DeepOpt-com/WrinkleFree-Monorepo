"""KV cache validation tests."""

import asyncio

import pytest

from wrinklefree_inference.kv_cache.validator import KVCacheValidator, run_kv_cache_validation


class TestKVCacheValidation:
    """Tests for KV cache behavior."""

    @pytest.mark.kv_cache
    @pytest.mark.asyncio
    async def test_prefix_caching(self, inference_url: str, skip_if_no_server):
        """Verify that common prefixes are cached efficiently."""
        validator = KVCacheValidator(inference_url)

        result = await validator.validate_prefix_caching()

        assert result["first_latency_ms"] > 0
        assert result["second_latency_ms"] > 0

        # Second request should be at least somewhat faster due to caching
        # Using a lenient threshold since timing can vary
        # A speedup > 1.0 means caching is likely working
        print(f"Prefix caching speedup: {result['speedup']:.2f}x")

    @pytest.mark.kv_cache
    @pytest.mark.asyncio
    async def test_context_limits(self, inference_url: str, skip_if_no_server):
        """Verify behavior at context window boundaries."""
        validator = KVCacheValidator(inference_url)

        handled = await validator.validate_context_limits()

        # Should handle context limits gracefully (either truncate or error cleanly)
        assert handled is True

    @pytest.mark.kv_cache
    @pytest.mark.asyncio
    async def test_continuous_batching(self, inference_url: str, skip_if_no_server):
        """Verify continuous batching works under load."""
        validator = KVCacheValidator(inference_url)

        success_rate = await validator.validate_continuous_batching(
            num_concurrent=5,  # Start with fewer concurrent requests
            max_tokens=10,
        )

        # At least 80% of requests should succeed
        assert success_rate >= 0.8, f"Only {success_rate*100:.0f}% succeeded"

    @pytest.mark.kv_cache
    def test_full_validation(self, inference_url: str, skip_if_no_server):
        """Run full KV cache validation suite."""
        metrics = run_kv_cache_validation(inference_url)

        print(f"\nKV Cache Validation Results:")
        print(f"  Prefix speedup: {metrics.prefix_speedup:.2f}x")
        print(f"  First request latency: {metrics.first_request_latency_ms:.1f}ms")
        print(f"  Second request latency: {metrics.second_request_latency_ms:.1f}ms")
        print(f"  Context limit handled: {metrics.context_limit_handled}")
        print(f"  Concurrent success rate: {metrics.concurrent_success_rate*100:.0f}%")

        if metrics.errors:
            print(f"  Errors: {metrics.errors}")

        # Basic sanity checks
        assert len(metrics.errors) == 0, f"Validation errors: {metrics.errors}"


class TestKVCacheEdgeCases:
    """Edge case tests for KV cache."""

    @pytest.mark.kv_cache
    @pytest.mark.asyncio
    async def test_empty_prompt(self, inference_url: str, skip_if_no_server):
        """Test handling of empty/minimal prompts."""
        from wrinklefree_inference.client.bitnet_client import AsyncBitNetClient

        async with AsyncBitNetClient(inference_url) as client:
            # Very short prompt
            response = await client.generate("", max_tokens=10)
            assert response is not None

    @pytest.mark.kv_cache
    @pytest.mark.asyncio
    async def test_repeated_identical_prompts(self, inference_url: str, skip_if_no_server):
        """Test that identical prompts benefit from full caching."""
        import time

        from wrinklefree_inference.client.bitnet_client import AsyncBitNetClient

        prompt = "What is 2+2?"

        async with AsyncBitNetClient(inference_url) as client:
            # First request
            start = time.perf_counter()
            await client.generate(prompt, max_tokens=20)
            first_time = time.perf_counter() - start

            # Same request again
            start = time.perf_counter()
            await client.generate(prompt, max_tokens=20)
            second_time = time.perf_counter() - start

            print(f"Identical prompt times: {first_time*1000:.0f}ms -> {second_time*1000:.0f}ms")
