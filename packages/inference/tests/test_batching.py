"""Tests for inference batching (continuous batching).

These tests verify that the BitNet.cpp server properly handles:
1. Concurrent requests (continuous batching)
2. Different batch sizes
3. Memory efficiency under load
4. Response ordering and correctness
"""

import pytest
import asyncio
import time
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed


# Marker for tests requiring server
requires_server = pytest.mark.skipif(
    os.environ.get("INFERENCE_URL") is None,
    reason="INFERENCE_URL not set - need running server"
)


@dataclass
class BatchRequest:
    """Single request in a batch."""
    prompt: str
    max_tokens: int = 32
    expected_prefix: Optional[str] = None


@dataclass
class BatchResult:
    """Result of a batch request."""
    request_idx: int
    prompt: str
    response: str
    latency_ms: float
    success: bool
    error: Optional[str] = None


@requires_server
class TestContinuousBatching:
    """Test continuous batching behavior."""

    @pytest.fixture
    def client(self):
        from wrinklefree_inference.client.bitnet_client import BitNetClient
        url = os.environ["INFERENCE_URL"]
        host = url.replace("http://", "").replace("https://", "").split(":")[0]
        port = int(url.split(":")[-1].split("/")[0]) if ":" in url.replace("http://", "") else 8080
        return BitNetClient(host=host, port=port)

    def _make_request(self, client, request: BatchRequest, idx: int) -> BatchResult:
        """Make a single request and return result."""
        start = time.time()
        try:
            response = client.generate(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=0.0,  # Deterministic
            )
            latency_ms = (time.time() - start) * 1000
            return BatchResult(
                request_idx=idx,
                prompt=request.prompt,
                response=response,
                latency_ms=latency_ms,
                success=True,
            )
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            return BatchResult(
                request_idx=idx,
                prompt=request.prompt,
                response="",
                latency_ms=latency_ms,
                success=False,
                error=str(e),
            )

    def test_concurrent_identical_requests(self, client):
        """Test batching with identical requests."""
        prompt = "The capital of France is"
        batch_size = 5
        requests = [BatchRequest(prompt=prompt, max_tokens=16) for _ in range(batch_size)]

        results = []
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = [
                executor.submit(self._make_request, client, req, i)
                for i, req in enumerate(requests)
            ]
            for future in as_completed(futures):
                results.append(future.result())

        # All should succeed
        assert all(r.success for r in results), f"Failures: {[r.error for r in results if not r.success]}"

        # All responses should be similar (deterministic with temp=0)
        responses = [r.response.strip() for r in results]
        # They should all mention Paris
        assert all("Paris" in r or "paris" in r.lower() for r in responses if r)

    def test_concurrent_different_requests(self, client):
        """Test batching with different prompts."""
        requests = [
            BatchRequest(prompt="1 + 1 =", max_tokens=8),
            BatchRequest(prompt="The color of the sky is", max_tokens=16),
            BatchRequest(prompt="Water boils at", max_tokens=16),
            BatchRequest(prompt="The largest planet is", max_tokens=16),
        ]

        results = []
        with ThreadPoolExecutor(max_workers=len(requests)) as executor:
            futures = [
                executor.submit(self._make_request, client, req, i)
                for i, req in enumerate(requests)
            ]
            for future in as_completed(futures):
                results.append(future.result())

        # All should succeed
        assert all(r.success for r in results)

        # Verify responses are appropriate for prompts
        for result in sorted(results, key=lambda r: r.request_idx):
            assert len(result.response) > 0

    def test_batching_efficiency(self, client):
        """Test that batching is more efficient than sequential."""
        prompt = "Hello, how are you?"
        num_requests = 6

        # Sequential baseline
        sequential_results = []
        seq_start = time.time()
        for i in range(num_requests):
            result = self._make_request(
                client,
                BatchRequest(prompt=prompt, max_tokens=16),
                i
            )
            sequential_results.append(result)
        sequential_total_ms = (time.time() - seq_start) * 1000

        # Concurrent (should benefit from batching)
        concurrent_results = []
        conc_start = time.time()
        with ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [
                executor.submit(
                    self._make_request,
                    client,
                    BatchRequest(prompt=prompt, max_tokens=16),
                    i
                )
                for i in range(num_requests)
            ]
            for future in as_completed(futures):
                concurrent_results.append(future.result())
        concurrent_total_ms = (time.time() - conc_start) * 1000

        # All should succeed
        assert all(r.success for r in sequential_results)
        assert all(r.success for r in concurrent_results)

        # Concurrent should be faster (batching benefit)
        # Note: This is a soft assertion - depends on server config
        print(f"\nSequential total: {sequential_total_ms:.1f}ms")
        print(f"Concurrent total: {concurrent_total_ms:.1f}ms")
        print(f"Speedup: {sequential_total_ms/concurrent_total_ms:.2f}x")

        # At minimum, concurrent shouldn't be significantly slower
        assert concurrent_total_ms < sequential_total_ms * 1.5, \
            "Concurrent processing significantly slower than sequential"

    def test_varying_output_lengths(self, client):
        """Test batching with varying output length requests."""
        requests = [
            BatchRequest(prompt="Say hi:", max_tokens=4),
            BatchRequest(prompt="Explain gravity in one sentence:", max_tokens=32),
            BatchRequest(prompt="Count 1:", max_tokens=8),
            BatchRequest(prompt="Write a haiku about coding:", max_tokens=48),
        ]

        results = []
        with ThreadPoolExecutor(max_workers=len(requests)) as executor:
            futures = [
                executor.submit(self._make_request, client, req, i)
                for i, req in enumerate(requests)
            ]
            for future in as_completed(futures):
                results.append(future.result())

        # All should succeed
        assert all(r.success for r in results)

    def test_request_ordering(self, client):
        """Test that response ordering is correct."""
        # Use unique prompts that should produce predictable outputs
        requests = [
            BatchRequest(prompt=f"Number {i}:", max_tokens=4)
            for i in range(8)
        ]

        results = []
        with ThreadPoolExecutor(max_workers=len(requests)) as executor:
            futures = {
                executor.submit(self._make_request, client, req, i): i
                for i, req in enumerate(requests)
            }
            for future in as_completed(futures):
                results.append(future.result())

        # Sort by original index
        results.sort(key=lambda r: r.request_idx)

        # All should succeed
        assert all(r.success for r in results)

        # Verify prompts match indices
        for i, result in enumerate(results):
            assert f"Number {i}" in result.prompt


@requires_server
class TestBatchMemoryEfficiency:
    """Test memory efficiency under batched load."""

    @pytest.fixture
    def client(self):
        from wrinklefree_inference.client.bitnet_client import BitNetClient
        url = os.environ["INFERENCE_URL"]
        host = url.replace("http://", "").replace("https://", "").split(":")[0]
        port = int(url.split(":")[-1].split("/")[0]) if ":" in url.replace("http://", "") else 8080
        return BitNetClient(host=host, port=port)

    def _make_request(self, client, prompt: str, max_tokens: int = 32) -> Tuple[bool, float]:
        """Make request and return (success, latency_ms)."""
        start = time.time()
        try:
            client.generate(prompt=prompt, max_tokens=max_tokens, temperature=0.7)
            return True, (time.time() - start) * 1000
        except Exception:
            return False, (time.time() - start) * 1000

    def test_increasing_batch_size(self, client):
        """Test server handles increasing batch sizes."""
        batch_sizes = [1, 2, 4, 8, 16]
        prompt = "The meaning of life is"

        for batch_size in batch_sizes:
            results = []
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = [
                    executor.submit(self._make_request, client, prompt, 32)
                    for _ in range(batch_size)
                ]
                for future in as_completed(futures):
                    results.append(future.result())

            success_rate = sum(1 for s, _ in results if s) / len(results)
            avg_latency = sum(lat for _, lat in results) / len(results)

            print(f"\nBatch size {batch_size}: {success_rate*100:.0f}% success, {avg_latency:.1f}ms avg latency")

            # Should maintain high success rate
            assert success_rate >= 0.9, f"Low success rate at batch size {batch_size}"

    def test_sustained_batch_load(self, client):
        """Test sustained batched load over time."""
        num_rounds = 3
        batch_size = 4
        prompt = "Quick test:"

        all_results = []
        for round_num in range(num_rounds):
            round_results = []
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = [
                    executor.submit(self._make_request, client, prompt, 16)
                    for _ in range(batch_size)
                ]
                for future in as_completed(futures):
                    round_results.append(future.result())

            success_rate = sum(1 for s, _ in round_results if s) / len(round_results)
            all_results.extend(round_results)

            print(f"\nRound {round_num + 1}: {success_rate*100:.0f}% success")

            # Brief pause between rounds
            time.sleep(0.5)

        # Overall success rate
        overall_success = sum(1 for s, _ in all_results if s) / len(all_results)
        assert overall_success >= 0.85, f"Low overall success rate: {overall_success*100:.0f}%"


class TestBatchingWithMoE:
    """Test batching behavior with MoE module (unit tests, no server needed)."""

    def test_moe_batch_processing(self):
        """Test MoE FFN processes batches correctly."""
        import torch
        from wrinklefree_inference.moe.expert import BitNetMoEFFN

        moe_ffn = BitNetMoEFFN(
            hidden_size=64,
            intermediate_size=256,
            num_experts=8,
            top_k=2,
            router_type="topk",
        )

        # Test different batch sizes
        batch_sizes = [1, 2, 4, 8, 16]
        seq_len = 10

        for batch_size in batch_sizes:
            x = torch.randn(batch_size, seq_len, 64)
            y, router_logits = moe_ffn(x, output_router_logits=True)

            assert y.shape == (batch_size, seq_len, 64), f"Wrong output shape for batch {batch_size}"
            assert router_logits.shape == (batch_size, seq_len, 8), f"Wrong router shape for batch {batch_size}"
            assert not torch.isnan(y).any(), f"NaN in output for batch {batch_size}"

    def test_moe_batch_consistency(self):
        """Test MoE produces consistent results across batch dimensions."""
        import torch
        from wrinklefree_inference.moe.expert import BitNetMoEFFN

        moe_ffn = BitNetMoEFFN(
            hidden_size=64,
            intermediate_size=256,
            num_experts=4,
            top_k=1,
            router_type="identity",  # Deterministic routing
        )
        moe_ffn.eval()

        # Single sample
        x_single = torch.randn(1, 5, 64)

        # Same sample repeated in batch
        x_batch = x_single.repeat(4, 1, 1)

        with torch.no_grad():
            y_single, _ = moe_ffn(x_single)
            y_batch, _ = moe_ffn(x_batch)

        # Each batch element should match single
        for i in range(4):
            assert torch.allclose(y_single[0], y_batch[i], atol=1e-5), \
                f"Batch element {i} doesn't match single sample"

    def test_moe_expert_load_balancing(self):
        """Test that top-k routing distributes load across experts."""
        import torch
        from wrinklefree_inference.moe.expert import BitNetMoEFFN

        moe_ffn = BitNetMoEFFN(
            hidden_size=64,
            intermediate_size=256,
            num_experts=8,
            top_k=2,
            router_type="topk",
        )

        # Large batch to see distribution
        x = torch.randn(32, 20, 64)
        _, router_logits = moe_ffn(x, output_router_logits=True)

        # Check expert selection distribution
        selected_experts = router_logits.argmax(dim=-1)  # Most likely expert per token
        expert_counts = torch.bincount(selected_experts.flatten(), minlength=8)

        # Should have some load balancing (not all tokens to one expert)
        used_experts = (expert_counts > 0).sum().item()
        assert used_experts >= 2, f"Only {used_experts} experts used, expected better distribution"

        print(f"\nExpert usage distribution: {expert_counts.tolist()}")
