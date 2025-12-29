"""Pytest configuration and fixtures for WrinkleFree inference tests."""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import pytest
import requests


@dataclass
class TestMetrics:
    """Collect metrics during test execution for AI agent consumption."""

    latencies_ms: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    requests_total: int = 0
    requests_failed: int = 0

    def record_latency(self, latency_ms: float) -> None:
        self.latencies_ms.append(latency_ms)
        self.requests_total += 1

    def record_error(self, error: str) -> None:
        self.errors.append(error)
        self.requests_failed += 1
        self.requests_total += 1

    def to_dict(self) -> dict[str, Any]:
        latencies = sorted(self.latencies_ms)
        return {
            "requests_total": self.requests_total,
            "requests_failed": self.requests_failed,
            "error_rate": self.requests_failed / max(self.requests_total, 1),
            "latency_p50_ms": latencies[len(latencies) // 2] if latencies else 0,
            "latency_p95_ms": latencies[int(len(latencies) * 0.95)] if latencies else 0,
            "latency_p99_ms": latencies[int(len(latencies) * 0.99)] if latencies else 0,
            "errors": self.errors[:10],  # First 10 errors
        }


def get_base_url() -> str:
    """Get inference server URL from environment or default."""
    return os.environ.get("INFERENCE_URL", "http://localhost:8080")


@pytest.fixture(scope="session")
def base_url() -> str:
    """Base URL for the inference server."""
    return get_base_url()


@pytest.fixture(scope="session")
def client(base_url: str) -> requests.Session:
    """HTTP client with retry configuration."""
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    return session


@pytest.fixture(scope="session")
def wait_for_healthy(base_url: str) -> None:
    """Wait for inference server to be healthy before running tests."""
    max_wait = int(os.environ.get("HEALTH_TIMEOUT", 120))
    start = time.time()

    while time.time() - start < max_wait:
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)

    pytest.fail(f"Inference server not healthy after {max_wait}s")


@pytest.fixture
def metrics() -> TestMetrics:
    """Collect metrics during test for reporting."""
    return TestMetrics()


# Store metrics globally for JSON report
_test_metrics: dict[str, dict] = {}


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Capture test metrics for JSON report."""
    outcome = yield
    report = outcome.get_result()

    if report.when == "call":
        # Get metrics from test if available
        if hasattr(item, "funcargs") and "metrics" in item.funcargs:
            metrics = item.funcargs["metrics"]
            _test_metrics[item.nodeid] = metrics.to_dict()


def pytest_sessionfinish(session, exitstatus):
    """Write aggregated metrics to file after test session."""
    if _test_metrics:
        metrics_file = os.environ.get("METRICS_FILE", "test_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(_test_metrics, f, indent=2)
