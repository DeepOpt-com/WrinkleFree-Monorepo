"""Locust load testing for WrinkleFree inference server.

Run interactively:
    locust -f tests/locustfile.py --host http://localhost:8080

Run headless (CI-friendly):
    locust -f tests/locustfile.py --host http://localhost:8080 \
        --headless -u 10 -r 2 -t 60s --json

With Docker Compose:
    docker compose -f docker-compose.test.yml --profile load up
"""

import json
import os
import time
from typing import Any

from locust import HttpUser, between, events, task


class InferenceUser(HttpUser):
    """Simulates a user making inference requests."""

    # Wait between 0.5 and 2 seconds between requests
    wait_time = between(0.5, 2.0)

    def on_start(self) -> None:
        """Wait for server to be healthy before starting."""
        self.wait_for_healthy()

    def wait_for_healthy(self, timeout: int = 60) -> None:
        """Wait for health endpoint to return 200."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                response = self.client.get("/health", timeout=5)
                if response.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(1)
        raise RuntimeError("Server not healthy after timeout")

    @task(10)
    def short_completion(self) -> None:
        """Most common: short prompt, few tokens."""
        self.client.post(
            "/v1/completions",
            json={
                "prompt": "Hello, how are you?",
                "max_tokens": 20,
                "temperature": 0.7,
            },
            name="/v1/completions (short)",
        )

    @task(5)
    def medium_completion(self) -> None:
        """Medium: moderate prompt and response."""
        self.client.post(
            "/v1/completions",
            json={
                "prompt": "Explain the concept of machine learning in simple terms. "
                "Focus on the key ideas and provide examples.",
                "max_tokens": 100,
                "temperature": 0.7,
            },
            name="/v1/completions (medium)",
        )

    @task(2)
    def long_completion(self) -> None:
        """Less common: longer generation."""
        self.client.post(
            "/v1/completions",
            json={
                "prompt": "Write a short story about a robot learning to paint.",
                "max_tokens": 200,
                "temperature": 0.8,
            },
            name="/v1/completions (long)",
        )

    @task(1)
    def health_check(self) -> None:
        """Periodic health checks (like load balancer would do)."""
        self.client.get("/health", name="/health")


class BurstUser(HttpUser):
    """Simulates burst traffic patterns."""

    wait_time = between(0.1, 0.5)  # Faster requests

    @task
    def rapid_short_completion(self) -> None:
        """Rapid-fire short completions."""
        self.client.post(
            "/v1/completions",
            json={
                "prompt": "Hi",
                "max_tokens": 5,
                "temperature": 0.0,
            },
            name="/v1/completions (burst)",
        )


# Metrics collection for JSON output
_custom_metrics: dict[str, Any] = {
    "requests_by_endpoint": {},
    "latency_percentiles": {},
    "errors_by_type": {},
}


@events.request.add_listener
def on_request(
    request_type: str,
    name: str,
    response_time: float,
    response_length: int,
    response: Any,
    exception: Any,
    **kwargs,
) -> None:
    """Collect custom metrics for each request."""
    if name not in _custom_metrics["requests_by_endpoint"]:
        _custom_metrics["requests_by_endpoint"][name] = {
            "count": 0,
            "failures": 0,
            "latencies": [],
        }

    endpoint_metrics = _custom_metrics["requests_by_endpoint"][name]
    endpoint_metrics["count"] += 1
    endpoint_metrics["latencies"].append(response_time)

    if exception:
        endpoint_metrics["failures"] += 1
        error_type = type(exception).__name__
        _custom_metrics["errors_by_type"][error_type] = (
            _custom_metrics["errors_by_type"].get(error_type, 0) + 1
        )


@events.quitting.add_listener
def on_quitting(environment: Any, **kwargs) -> None:
    """Write custom metrics to file on exit."""
    # Calculate percentiles
    for name, metrics in _custom_metrics["requests_by_endpoint"].items():
        latencies = sorted(metrics["latencies"])
        if latencies:
            _custom_metrics["latency_percentiles"][name] = {
                "p50": latencies[len(latencies) // 2],
                "p95": latencies[int(len(latencies) * 0.95)],
                "p99": latencies[int(len(latencies) * 0.99)],
                "min": latencies[0],
                "max": latencies[-1],
            }
        # Remove raw latencies to reduce output size
        del metrics["latencies"]

    # Write to file if specified
    output_file = os.environ.get("LOCUST_METRICS_FILE", "locust_metrics.json")
    with open(output_file, "w") as f:
        json.dump(_custom_metrics, f, indent=2)
    print(f"Custom metrics written to {output_file}")
