"""Smoke tests for basic inference server functionality.

These tests validate that the server is running and responds correctly.
They should pass quickly (<10s total) and are suitable for CI/CD gates.

Run with:
    uv run pytest tests/test_smoke.py -v
"""

import time

import pytest
import requests


@pytest.mark.smoke
def test_health_endpoint(base_url: str, wait_for_healthy: None) -> None:
    """Health check endpoint returns 200."""
    response = requests.get(f"{base_url}/health", timeout=10)
    assert response.status_code == 200, f"Health check failed: {response.text}"


@pytest.mark.smoke
def test_completions_endpoint_exists(base_url: str, wait_for_healthy: None) -> None:
    """Completions endpoint is available."""
    response = requests.post(
        f"{base_url}/v1/completions",
        json={"prompt": "Hi", "max_tokens": 1},
        timeout=30,
    )
    # Should return 200 or 400 (bad request), not 404
    assert response.status_code != 404, "Completions endpoint not found"


@pytest.mark.smoke
def test_basic_completion(base_url: str, wait_for_healthy: None) -> None:
    """Basic completion request works."""
    response = requests.post(
        f"{base_url}/v1/completions",
        json={
            "prompt": "The capital of France is",
            "max_tokens": 10,
            "temperature": 0.0,
        },
        timeout=60,
    )
    assert response.status_code == 200, f"Completion failed: {response.text}"

    data = response.json()
    assert "choices" in data, f"Response missing 'choices': {data}"
    assert len(data["choices"]) > 0, "No choices returned"
    assert "text" in data["choices"][0], f"Choice missing 'text': {data}"


@pytest.mark.smoke
def test_completion_response_schema(base_url: str, wait_for_healthy: None) -> None:
    """Completion response matches OpenAI schema."""
    response = requests.post(
        f"{base_url}/v1/completions",
        json={"prompt": "Hello", "max_tokens": 5},
        timeout=60,
    )
    assert response.status_code == 200

    data = response.json()

    # Required fields per OpenAI spec
    required_fields = ["id", "object", "created", "model", "choices"]
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"

    # Validate choices structure
    assert isinstance(data["choices"], list)
    if data["choices"]:
        choice = data["choices"][0]
        assert "text" in choice or "message" in choice
        assert "index" in choice


@pytest.mark.smoke
def test_completion_latency_acceptable(
    base_url: str, wait_for_healthy: None, metrics
) -> None:
    """First token latency is under threshold."""
    threshold_ms = int(pytest.config.getoption("--latency-threshold", default=5000))

    start = time.time()
    response = requests.post(
        f"{base_url}/v1/completions",
        json={"prompt": "Count to 5:", "max_tokens": 20},
        timeout=60,
    )
    latency_ms = (time.time() - start) * 1000

    metrics.record_latency(latency_ms)

    assert response.status_code == 200
    assert latency_ms < threshold_ms, f"Latency {latency_ms:.0f}ms exceeds {threshold_ms}ms threshold"


@pytest.mark.smoke
def test_invalid_request_returns_error(base_url: str, wait_for_healthy: None) -> None:
    """Invalid request returns appropriate error."""
    response = requests.post(
        f"{base_url}/v1/completions",
        json={"invalid_field": "value"},  # Missing required 'prompt'
        timeout=30,
    )
    # Should return 400 or 422, not 500
    assert response.status_code in [400, 422], f"Expected 4xx, got {response.status_code}"


@pytest.mark.smoke
def test_empty_prompt_handled(base_url: str, wait_for_healthy: None) -> None:
    """Empty prompt is handled gracefully."""
    response = requests.post(
        f"{base_url}/v1/completions",
        json={"prompt": "", "max_tokens": 5},
        timeout=30,
    )
    # Should either work or return 4xx, not crash
    assert response.status_code in [200, 400, 422]


def pytest_addoption(parser):
    """Add custom CLI options."""
    parser.addoption(
        "--latency-threshold",
        action="store",
        default=5000,
        help="Maximum acceptable latency in milliseconds",
    )
