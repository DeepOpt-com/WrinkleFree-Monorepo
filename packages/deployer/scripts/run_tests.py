#!/usr/bin/env python3
"""Test runner with structured JSON output for AI agent consumption.

Usage:
    uv run python scripts/run_tests.py --suite smoke --output json
    uv run python scripts/run_tests.py --suite all --output json
    uv run python scripts/run_tests.py --suite load --duration 60 --output json

Exit codes:
    0 - All tests passed
    1 - Test failures
    2 - Infrastructure error (service not reachable)
    3 - Configuration error
"""

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import click
import requests


@dataclass
class TestResult:
    """Individual test result."""

    name: str
    status: str  # "passed", "failed", "skipped", "error"
    duration_seconds: float
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSummary:
    """Overall test run summary."""

    suite: str
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    duration_seconds: float = 0.0
    tests: list[TestResult] = field(default_factory=list)
    load_test: dict[str, Any] | None = None
    infrastructure: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["tests"] = [asdict(t) for t in self.tests]
        return result


def check_infrastructure(base_url: str, timeout: int = 10) -> dict[str, Any]:
    """Check if inference server is reachable."""
    info = {
        "url": base_url,
        "healthy": False,
        "response_time_ms": None,
        "error": None,
    }

    try:
        start = time.time()
        response = requests.get(f"{base_url}/health", timeout=timeout)
        info["response_time_ms"] = (time.time() - start) * 1000
        info["healthy"] = response.status_code == 200
        if not info["healthy"]:
            info["error"] = f"Health check returned {response.status_code}"
    except requests.RequestException as e:
        info["error"] = str(e)

    return info


def run_pytest(
    suite: str,
    base_url: str,
    extra_args: list[str] | None = None,
) -> tuple[int, TestSummary]:
    """Run pytest and parse results."""
    summary = TestSummary(suite=suite)

    # Determine which tests to run
    test_path = Path(__file__).parent.parent / "tests"
    if suite == "smoke":
        test_args = [str(test_path / "test_smoke.py"), "-m", "smoke"]
    elif suite == "integration":
        test_args = [str(test_path / "test_integration.py"), "-m", "integration"]
    elif suite == "all":
        test_args = [str(test_path)]
    else:
        test_args = [str(test_path / f"test_{suite}.py")]

    # Build pytest command
    json_report = Path("/tmp/pytest_report.json")
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *test_args,
        "--json-report",
        f"--json-report-file={json_report}",
        "-v",
    ]
    if extra_args:
        cmd.extend(extra_args)

    # Set environment
    env = os.environ.copy()
    env["INFERENCE_URL"] = base_url

    # Run pytest
    start = time.time()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    summary.duration_seconds = time.time() - start

    # Parse JSON report
    if json_report.exists():
        with open(json_report) as f:
            report = json.load(f)

        summary.total = report.get("summary", {}).get("total", 0)
        summary.passed = report.get("summary", {}).get("passed", 0)
        summary.failed = report.get("summary", {}).get("failed", 0)
        summary.skipped = report.get("summary", {}).get("skipped", 0)
        summary.errors = report.get("summary", {}).get("error", 0)

        # Extract individual test results
        for test in report.get("tests", []):
            test_result = TestResult(
                name=test.get("nodeid", "unknown"),
                status=test.get("outcome", "unknown"),
                duration_seconds=test.get("duration", 0),
            )
            if test.get("outcome") == "failed":
                call = test.get("call", {})
                test_result.error = call.get("longrepr", "Unknown error")
            summary.tests.append(test_result)

    return result.returncode, summary


def run_cloud_smoke(
    cloud: str,
    output: str,
) -> tuple[int, TestSummary]:
    """Run cloud smoke tests for spot instance validation."""
    summary = TestSummary(suite=f"cloud-{cloud}")

    test_path = Path(__file__).parent.parent / "tests" / "test_cloud_smoke.py"
    json_report = Path("/tmp/pytest_cloud_report.json")
    results_file = Path("/tmp/cloud_smoke_results.json")

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(test_path),
        f"--cloud={cloud}",
        "-m", "cloud",  # Only run cloud-marked tests
        "--json-report",
        f"--json-report-file={json_report}",
        "-v",
    ]

    env = os.environ.copy()
    env["CLOUD_SMOKE_RESULTS"] = str(results_file)

    # Run pytest
    start = time.time()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    summary.duration_seconds = time.time() - start

    # Parse JSON report
    if json_report.exists():
        with open(json_report) as f:
            report = json.load(f)

        summary.total = report.get("summary", {}).get("total", 0)
        summary.passed = report.get("summary", {}).get("passed", 0)
        summary.failed = report.get("summary", {}).get("failed", 0)
        summary.skipped = report.get("summary", {}).get("skipped", 0)

        for test in report.get("tests", []):
            test_result = TestResult(
                name=test.get("nodeid", "unknown"),
                status=test.get("outcome", "unknown"),
                duration_seconds=test.get("duration", 0),
            )
            if test.get("outcome") == "failed":
                call = test.get("call", {})
                test_result.error = call.get("longrepr", "Unknown error")
            summary.tests.append(test_result)

    # Load cloud-specific results if available
    if results_file.exists():
        with open(results_file) as f:
            cloud_results = json.load(f)
        summary.infrastructure = {"cloud": cloud, "results": cloud_results}

    # Output
    if output == "json":
        click.echo(json.dumps(summary.to_dict(), indent=2))
    else:
        click.echo(f"\n{'='*60}")
        click.echo(f"Cloud Smoke Test: {cloud.upper()}")
        click.echo(f"{'='*60}")
        click.echo(f"Total: {summary.total}")
        click.echo(f"Passed: {summary.passed}")
        click.echo(f"Failed: {summary.failed}")
        click.echo(f"Duration: {summary.duration_seconds:.1f}s")

        if summary.failed > 0:
            click.echo(f"\nFailed tests:")
            for test in summary.tests:
                if test.status == "failed":
                    click.echo(f"  - {test.name}")

    return result.returncode, summary


def run_load_test(
    base_url: str,
    duration: int,
    users: int,
    spawn_rate: int,
) -> tuple[int, dict[str, Any]]:
    """Run locust load test and return metrics."""
    locust_file = Path(__file__).parent.parent / "tests" / "locustfile.py"
    metrics_file = Path("/tmp/locust_metrics.json")

    cmd = [
        sys.executable,
        "-m",
        "locust",
        "-f",
        str(locust_file),
        "--host",
        base_url,
        "--headless",
        "-u",
        str(users),
        "-r",
        str(spawn_rate),
        "-t",
        f"{duration}s",
        "--json",
    ]

    env = os.environ.copy()
    env["LOCUST_METRICS_FILE"] = str(metrics_file)

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    metrics: dict[str, Any] = {
        "duration_seconds": duration,
        "users": users,
        "spawn_rate": spawn_rate,
    }

    # Parse locust JSON output
    try:
        if result.stdout:
            locust_data = json.loads(result.stdout)
            if locust_data:
                # Extract aggregate stats
                for stat in locust_data:
                    if stat.get("name") == "Aggregated":
                        metrics["requests_total"] = stat.get("num_requests", 0)
                        metrics["requests_failed"] = stat.get("num_failures", 0)
                        metrics["requests_per_second"] = stat.get("current_rps", 0)
                        metrics["latency_avg_ms"] = stat.get("avg_response_time", 0)
                        metrics["latency_p50_ms"] = stat.get("response_times", {}).get("50", 0)
                        metrics["latency_p95_ms"] = stat.get("response_times", {}).get("95", 0)
                        metrics["latency_p99_ms"] = stat.get("response_times", {}).get("99", 0)
    except json.JSONDecodeError:
        metrics["parse_error"] = "Failed to parse locust output"

    # Load custom metrics if available
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics["custom"] = json.load(f)

    return result.returncode, metrics


@click.command()
@click.option(
    "--suite",
    type=click.Choice(["smoke", "integration", "load", "cloud", "all"]),
    default="smoke",
    help="Test suite to run",
)
@click.option(
    "--url",
    default="http://localhost:8080",
    help="Inference server URL",
)
@click.option(
    "--output",
    type=click.Choice(["json", "text"]),
    default="text",
    help="Output format",
)
@click.option(
    "--duration",
    default=60,
    help="Load test duration in seconds",
)
@click.option(
    "--users",
    default=10,
    help="Number of concurrent users for load test",
)
@click.option(
    "--spawn-rate",
    default=2,
    help="User spawn rate for load test",
)
@click.option(
    "--cloud",
    type=click.Choice(["aws", "gcp", "azure"]),
    default="aws",
    help="Cloud provider for cloud smoke tests",
)
def main(
    suite: str,
    url: str,
    output: str,
    duration: int,
    users: int,
    spawn_rate: int,
    cloud: str,
) -> None:
    """Run WrinkleFree inference tests with structured output."""
    # Cloud tests don't need local infrastructure
    if suite == "cloud":
        exit_code, result = run_cloud_smoke(cloud, output)
        sys.exit(exit_code)

    # Check infrastructure first
    infra = check_infrastructure(url)
    if not infra["healthy"]:
        result = TestSummary(
            suite=suite,
            infrastructure=infra,
        )
        if output == "json":
            click.echo(json.dumps(result.to_dict(), indent=2))
        else:
            click.echo(f"ERROR: Infrastructure not healthy: {infra['error']}")
        sys.exit(2)

    # Run appropriate tests
    if suite == "load":
        exit_code, load_metrics = run_load_test(url, duration, users, spawn_rate)
        result = TestSummary(
            suite=suite,
            load_test=load_metrics,
            infrastructure=infra,
        )
        # Determine pass/fail based on error rate
        error_rate = load_metrics.get("requests_failed", 0) / max(
            load_metrics.get("requests_total", 1), 1
        )
        if error_rate > 0.05:  # >5% error rate = failure
            exit_code = 1
    else:
        exit_code, result = run_pytest(suite, url)
        result.infrastructure = infra

    # Output results
    if output == "json":
        click.echo(json.dumps(result.to_dict(), indent=2))
    else:
        click.echo(f"\n{'='*60}")
        click.echo(f"Test Suite: {suite}")
        click.echo(f"{'='*60}")
        click.echo(f"Total: {result.total}")
        click.echo(f"Passed: {result.passed}")
        click.echo(f"Failed: {result.failed}")
        click.echo(f"Duration: {result.duration_seconds:.1f}s")

        if result.failed > 0:
            click.echo(f"\nFailed tests:")
            for test in result.tests:
                if test.status == "failed":
                    click.echo(f"  - {test.name}")
                    if test.error:
                        click.echo(f"    {test.error[:200]}...")

        if result.load_test:
            click.echo(f"\nLoad Test Results:")
            click.echo(f"  RPS: {result.load_test.get('requests_per_second', 'N/A')}")
            click.echo(f"  P95 Latency: {result.load_test.get('latency_p95_ms', 'N/A')}ms")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
