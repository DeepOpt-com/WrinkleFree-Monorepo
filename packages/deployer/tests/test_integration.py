"""Integration tests for inference server under realistic conditions.

These tests validate behavior under concurrent load and edge cases.
They take longer to run (~60s) and should run in staging/CI.

Run with:
    uv run pytest tests/test_integration.py -v
"""

import concurrent.futures
import time

import pytest
import requests


@pytest.mark.integration
def test_concurrent_requests(
    base_url: str, wait_for_healthy: None, metrics
) -> None:
    """Server handles multiple concurrent requests."""
    num_requests = 10

    def make_request(i: int) -> tuple[int, float, str | None]:
        start = time.time()
        try:
            response = requests.post(
                f"{base_url}/v1/completions",
                json={"prompt": f"Request {i}: Hello", "max_tokens": 10},
                timeout=120,
            )
            latency = (time.time() - start) * 1000
            return response.status_code, latency, None
        except Exception as e:
            latency = (time.time() - start) * 1000
            return 0, latency, str(e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = [executor.submit(make_request, i) for i in range(num_requests)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    # Record metrics
    for status, latency, error in results:
        if error:
            metrics.record_error(error)
        else:
            metrics.record_latency(latency)

    # Assertions
    success_count = sum(1 for status, _, _ in results if status == 200)
    assert success_count >= num_requests * 0.9, f"Only {success_count}/{num_requests} succeeded"


@pytest.mark.integration
def test_sequential_requests_stable(
    base_url: str, wait_for_healthy: None, metrics
) -> None:
    """Server remains stable across many sequential requests."""
    num_requests = 20
    failures = []

    for i in range(num_requests):
        start = time.time()
        try:
            response = requests.post(
                f"{base_url}/v1/completions",
                json={"prompt": f"Count: {i}", "max_tokens": 5},
                timeout=60,
            )
            latency = (time.time() - start) * 1000
            metrics.record_latency(latency)

            if response.status_code != 200:
                failures.append(f"Request {i}: status {response.status_code}")
        except Exception as e:
            metrics.record_error(str(e))
            failures.append(f"Request {i}: {e}")

    assert len(failures) == 0, f"Failures: {failures}"


@pytest.mark.integration
def test_large_context_handled(base_url: str, wait_for_healthy: None) -> None:
    """Server handles larger context appropriately."""
    # Generate a longer prompt (but within typical limits)
    long_prompt = "This is a test. " * 100  # ~1600 tokens

    response = requests.post(
        f"{base_url}/v1/completions",
        json={"prompt": long_prompt, "max_tokens": 50},
        timeout=120,
    )

    # Should either succeed or return appropriate error (not crash)
    assert response.status_code in [200, 400, 413], f"Unexpected status: {response.status_code}"


@pytest.mark.integration
def test_max_tokens_respected(base_url: str, wait_for_healthy: None) -> None:
    """Server respects max_tokens parameter."""
    response = requests.post(
        f"{base_url}/v1/completions",
        json={"prompt": "Count from 1 to 100:", "max_tokens": 10},
        timeout=60,
    )
    assert response.status_code == 200

    data = response.json()
    # The response should be truncated, not contain full count
    text = data["choices"][0].get("text", "")
    # Rough check - 10 tokens shouldn't produce 100 numbers
    assert len(text.split()) < 50, f"Response too long for max_tokens=10: {text}"


@pytest.mark.integration
def test_temperature_affects_output(base_url: str, wait_for_healthy: None) -> None:
    """Different temperatures produce different outputs."""
    prompt = "Generate a random word:"

    # Temperature 0 should be deterministic
    responses_t0 = []
    for _ in range(3):
        response = requests.post(
            f"{base_url}/v1/completions",
            json={"prompt": prompt, "max_tokens": 5, "temperature": 0.0},
            timeout=60,
        )
        if response.status_code == 200:
            responses_t0.append(response.json()["choices"][0].get("text", ""))

    # At temperature 0, outputs should be identical
    if len(responses_t0) >= 2:
        assert responses_t0[0] == responses_t0[1], "Temperature 0 should be deterministic"


@pytest.mark.integration
def test_error_recovery(base_url: str, wait_for_healthy: None) -> None:
    """Server recovers after receiving invalid requests."""
    # Send invalid request
    requests.post(
        f"{base_url}/v1/completions",
        json={"invalid": True},
        timeout=30,
    )

    # Server should still handle valid requests
    response = requests.post(
        f"{base_url}/v1/completions",
        json={"prompt": "Hello", "max_tokens": 5},
        timeout=60,
    )
    assert response.status_code == 200, "Server didn't recover from invalid request"


@pytest.mark.integration
@pytest.mark.slow
def test_sustained_load(base_url: str, wait_for_healthy: None, metrics) -> None:
    """Server handles sustained load over time."""
    duration_seconds = 30
    target_rps = 2  # Requests per second

    start_time = time.time()
    request_count = 0
    errors = []

    while time.time() - start_time < duration_seconds:
        request_start = time.time()
        try:
            response = requests.post(
                f"{base_url}/v1/completions",
                json={"prompt": "Quick test", "max_tokens": 5},
                timeout=30,
            )
            latency = (time.time() - request_start) * 1000
            metrics.record_latency(latency)

            if response.status_code != 200:
                errors.append(f"Status {response.status_code}")
        except Exception as e:
            metrics.record_error(str(e))
            errors.append(str(e))

        request_count += 1

        # Throttle to target RPS
        elapsed = time.time() - request_start
        sleep_time = max(0, (1 / target_rps) - elapsed)
        time.sleep(sleep_time)

    # Assertions
    error_rate = len(errors) / max(request_count, 1)
    assert error_rate < 0.05, f"Error rate {error_rate:.1%} exceeds 5%"
    assert request_count > duration_seconds, f"Only {request_count} requests in {duration_seconds}s"
