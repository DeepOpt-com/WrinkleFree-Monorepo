"""Smoke tests for inference engine."""

import pytest

from wf_infer.client.bitnet_client import BitNetClient


class TestSmoke:
    """Basic smoke tests that verify core functionality."""

    @pytest.mark.smoke
    def test_client_creation(self):
        """Test that client can be instantiated."""
        client = BitNetClient(host="localhost", port=8080)
        assert client.base_url == "http://localhost:8080"

    @pytest.mark.smoke
    def test_client_from_env(self, inference_url: str):
        """Test client configuration from environment."""
        assert inference_url is not None
        assert inference_url.startswith("http")


class TestIntegration:
    """Integration tests requiring a running server."""

    @pytest.mark.integration
    def test_health_check(self, client: BitNetClient, skip_if_no_server):
        """Test server health check endpoint."""
        assert client.health_check() is True

    @pytest.mark.integration
    def test_generate_simple(self, client: BitNetClient, skip_if_no_server):
        """Test basic text generation."""
        response = client.generate(
            prompt="The capital of France is",
            max_tokens=10,
            temperature=0.0,  # Deterministic
        )
        assert response is not None
        assert len(response) > 0

    @pytest.mark.integration
    def test_generate_with_stop(self, client: BitNetClient, skip_if_no_server):
        """Test generation with stop sequences."""
        response = client.generate(
            prompt="Count: 1, 2, 3,",
            max_tokens=50,
            stop=["\n", "."],
        )
        assert response is not None
        # Should stop at first newline or period
        assert "\n" not in response or "." not in response

    @pytest.mark.integration
    def test_chat_completion(self, client: BitNetClient, skip_if_no_server):
        """Test chat-style completion."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello in one word."},
        ]
        response = client.chat(messages, max_tokens=10)
        assert response is not None
        assert len(response) > 0

    @pytest.mark.integration
    def test_tokenize_detokenize(self, client: BitNetClient, skip_if_no_server):
        """Test tokenization round-trip."""
        text = "Hello, world!"
        tokens = client.tokenize(text)
        assert isinstance(tokens, list)
        assert len(tokens) > 0

        # Detokenize
        recovered = client.detokenize(tokens)
        assert text in recovered or recovered in text  # May have minor differences

    @pytest.mark.integration
    def test_streaming(self, client: BitNetClient, skip_if_no_server):
        """Test streaming generation."""
        chunks = list(client.generate_stream(
            prompt="Once upon a time",
            max_tokens=20,
        ))
        assert len(chunks) > 0
        full_text = "".join(chunks)
        assert len(full_text) > 0


class TestLatency:
    """Latency benchmarks for monitoring performance."""

    @pytest.mark.integration
    def test_first_token_latency(self, client: BitNetClient, skip_if_no_server):
        """Measure time to first token."""
        import time

        start = time.perf_counter()
        for chunk in client.generate_stream("Hello", max_tokens=5):
            first_token_latency = time.perf_counter() - start
            break

        # Should get first token within 5 seconds on CPU
        assert first_token_latency < 5.0, f"First token took {first_token_latency:.2f}s"

    @pytest.mark.integration
    def test_throughput(self, client: BitNetClient, skip_if_no_server):
        """Measure generation throughput."""
        import time

        prompt = "Write a short story about"
        max_tokens = 50

        start = time.perf_counter()
        response = client.generate(prompt, max_tokens=max_tokens)
        elapsed = time.perf_counter() - start

        # Estimate tokens generated (rough)
        tokens_generated = len(response.split())
        tokens_per_second = tokens_generated / elapsed if elapsed > 0 else 0

        # Minimum throughput expectation for CPU
        assert tokens_per_second > 1, f"Throughput too low: {tokens_per_second:.1f} tok/s"
