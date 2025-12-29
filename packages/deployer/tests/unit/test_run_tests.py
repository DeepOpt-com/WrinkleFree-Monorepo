"""
Unit tests for scripts/run_tests.py

These tests mock external dependencies to test the script logic
without requiring running services.
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add scripts to path for import
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from run_tests import (
    TestResult,
    TestSummary,
    check_infrastructure,
    run_pytest,
    run_cloud_smoke,
    run_load_test,
)


class TestTestResult:
    """Tests for TestResult dataclass."""

    def test_create_passed_result(self):
        """Can create a passed test result."""
        result = TestResult(
            name="test_example",
            status="passed",
            duration_seconds=1.5,
        )
        assert result.name == "test_example"
        assert result.status == "passed"
        assert result.duration_seconds == 1.5
        assert result.error is None

    def test_create_failed_result(self):
        """Can create a failed test result with error."""
        result = TestResult(
            name="test_failing",
            status="failed",
            duration_seconds=0.5,
            error="AssertionError: expected 1, got 2",
        )
        assert result.status == "failed"
        assert "AssertionError" in result.error

    def test_result_with_metrics(self):
        """Can attach metrics to test result."""
        result = TestResult(
            name="test_latency",
            status="passed",
            duration_seconds=2.0,
            metrics={"p50_ms": 100, "p99_ms": 250},
        )
        assert result.metrics["p50_ms"] == 100
        assert result.metrics["p99_ms"] == 250


class TestTestSummary:
    """Tests for TestSummary dataclass."""

    def test_create_empty_summary(self):
        """Can create empty summary."""
        summary = TestSummary(suite="smoke")
        assert summary.suite == "smoke"
        assert summary.total == 0
        assert summary.passed == 0
        assert summary.failed == 0

    def test_summary_to_dict(self):
        """Summary can be converted to dict."""
        summary = TestSummary(
            suite="integration",
            total=10,
            passed=8,
            failed=2,
            duration_seconds=45.5,
        )
        result = summary.to_dict()
        assert result["suite"] == "integration"
        assert result["total"] == 10
        assert result["passed"] == 8
        assert result["failed"] == 2

    def test_summary_with_tests(self):
        """Summary includes test results."""
        test1 = TestResult(name="test_a", status="passed", duration_seconds=1.0)
        test2 = TestResult(name="test_b", status="failed", duration_seconds=2.0, error="Error")

        summary = TestSummary(
            suite="smoke",
            total=2,
            passed=1,
            failed=1,
            tests=[test1, test2],
        )
        result = summary.to_dict()
        assert len(result["tests"]) == 2
        assert result["tests"][0]["name"] == "test_a"
        assert result["tests"][1]["status"] == "failed"


class TestCheckInfrastructure:
    """Tests for check_infrastructure function."""

    @patch("run_tests.requests.get")
    def test_healthy_server(self, mock_get):
        """Returns healthy status for 200 response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        result = check_infrastructure("http://localhost:8080", timeout=5)

        assert result["healthy"] is True
        assert result["url"] == "http://localhost:8080"
        assert result["response_time_ms"] is not None
        assert result["error"] is None

    @patch("run_tests.requests.get")
    def test_unhealthy_server(self, mock_get):
        """Returns unhealthy status for non-200 response."""
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_get.return_value = mock_response

        result = check_infrastructure("http://localhost:8080")

        assert result["healthy"] is False
        assert "503" in result["error"]

    @patch("run_tests.requests.get")
    def test_connection_error(self, mock_get):
        """Handles connection errors gracefully."""
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

        result = check_infrastructure("http://localhost:8080")

        assert result["healthy"] is False
        assert result["error"] is not None
        assert "Connection" in result["error"]

    @patch("run_tests.requests.get")
    def test_timeout_error(self, mock_get):
        """Handles timeout errors gracefully."""
        import requests
        mock_get.side_effect = requests.exceptions.Timeout("Request timed out")

        result = check_infrastructure("http://localhost:8080")

        assert result["healthy"] is False
        assert "timed out" in result["error"].lower() or "Timeout" in result["error"]


class TestRunPytest:
    """Tests for run_pytest function."""

    @patch("run_tests.subprocess.run")
    def test_smoke_suite_uses_correct_path(self, mock_run):
        """Smoke suite runs correct test file."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        # Create a mock JSON report
        with patch("builtins.open", MagicMock()):
            with patch.object(Path, "exists", return_value=False):
                exit_code, summary = run_pytest("smoke", "http://localhost:8080")

        # Check that subprocess was called
        assert mock_run.called
        call_args = mock_run.call_args[0][0]
        assert "test_smoke.py" in str(call_args)
        assert "-m" in call_args
        assert "smoke" in call_args

    @patch("run_tests.subprocess.run")
    def test_all_suite_runs_all_tests(self, mock_run):
        """All suite runs the tests directory."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        with patch.object(Path, "exists", return_value=False):
            exit_code, summary = run_pytest("all", "http://localhost:8080")

        call_args = mock_run.call_args[0][0]
        # Should include tests directory, not a specific file
        assert any("tests" in str(arg) and "test_" not in str(arg) for arg in call_args)

    @patch("run_tests.subprocess.run")
    def test_parses_json_report(self, mock_run):
        """Parses pytest JSON report correctly."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        mock_report = {
            "summary": {
                "total": 5,
                "passed": 4,
                "failed": 1,
                "skipped": 0,
            },
            "tests": [
                {"nodeid": "test_a", "outcome": "passed", "duration": 1.0},
                {"nodeid": "test_b", "outcome": "failed", "duration": 2.0, "call": {"longrepr": "Error"}},
            ],
        }

        with patch("builtins.open", MagicMock(return_value=MagicMock(
            __enter__=MagicMock(return_value=MagicMock(read=MagicMock(return_value=json.dumps(mock_report)))),
            __exit__=MagicMock(return_value=False),
        ))):
            with patch.object(Path, "exists", return_value=True):
                with patch("json.load", return_value=mock_report):
                    exit_code, summary = run_pytest("smoke", "http://localhost:8080")

        assert summary.total == 5
        assert summary.passed == 4
        assert summary.failed == 1


class TestErrorRateCalculation:
    """Tests for error rate logic."""

    def test_zero_requests_no_division_error(self):
        """Handles zero requests without division by zero."""
        # This tests the logic in main() for load test error rate
        requests_total = 0
        requests_failed = 0

        # Should not raise ZeroDivisionError
        error_rate = requests_failed / max(requests_total, 1)
        assert error_rate == 0.0

    def test_high_error_rate_fails(self):
        """Error rate > 5% should fail load test."""
        requests_total = 100
        requests_failed = 10

        error_rate = requests_failed / max(requests_total, 1)
        assert error_rate == 0.1
        assert error_rate > 0.05  # Threshold for failure


class TestRunCloudSmoke:
    """Tests for run_cloud_smoke function.

    This function runs cloud smoke tests using pytest and parses the results.
    We test the result parsing logic without actually running cloud tests.
    """

    @patch("run_tests.subprocess.run")
    @patch("run_tests.click.echo")
    def test_cloud_smoke_parses_json_report(self, mock_echo, mock_run):
        """Parses pytest JSON report correctly for cloud tests."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        mock_report = {
            "summary": {"total": 3, "passed": 2, "failed": 1, "skipped": 0},
            "tests": [
                {"nodeid": "test_aws::test_launch", "outcome": "passed", "duration": 30.0},
                {"nodeid": "test_aws::test_health", "outcome": "passed", "duration": 5.0},
                {"nodeid": "test_aws::test_inference", "outcome": "failed", "duration": 10.0,
                 "call": {"longrepr": "Connection refused"}},
            ],
        }

        with patch.object(Path, "exists", return_value=True):
            with patch("builtins.open", MagicMock()):
                with patch("json.load", return_value=mock_report):
                    exit_code, summary = run_cloud_smoke("aws", "json")

        assert summary.suite == "cloud-aws"
        assert summary.total == 3
        assert summary.passed == 2
        assert summary.failed == 1
        assert len(summary.tests) == 3

    @patch("run_tests.subprocess.run")
    @patch("run_tests.click.echo")
    def test_cloud_smoke_text_output_shows_failures(self, mock_echo, mock_run):
        """Text output displays failed tests prominently."""
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")

        mock_report = {
            "summary": {"total": 2, "passed": 1, "failed": 1},
            "tests": [
                {"nodeid": "test_gcp::test_spot", "outcome": "passed", "duration": 15.0},
                {"nodeid": "test_gcp::test_preempt", "outcome": "failed", "duration": 5.0,
                 "call": {"longrepr": "Instance preempted"}},
            ],
        }

        with patch.object(Path, "exists", return_value=True):
            with patch("builtins.open", MagicMock()):
                with patch("json.load", return_value=mock_report):
                    exit_code, summary = run_cloud_smoke("gcp", "text")

        # Should call click.echo multiple times for text output
        assert mock_echo.called
        assert summary.failed == 1

    @patch("run_tests.subprocess.run")
    @patch("run_tests.click.echo")
    def test_cloud_smoke_loads_cloud_results_file(self, mock_echo, mock_run):
        """Loads cloud-specific results when available."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        mock_report = {"summary": {"total": 1, "passed": 1}, "tests": []}
        mock_cloud_results = {
            "instance_id": "i-12345",
            "launch_time_seconds": 45.2,
            "spot_price": 0.03,
        }

        # Track which file is being opened
        call_count = [0]
        def mock_json_load(f):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_report
            return mock_cloud_results

        with patch.object(Path, "exists", return_value=True):
            with patch("builtins.open", MagicMock()):
                with patch("json.load", side_effect=mock_json_load):
                    exit_code, summary = run_cloud_smoke("aws", "json")

        assert "cloud" in summary.infrastructure
        assert summary.infrastructure["cloud"] == "aws"

    @patch("run_tests.subprocess.run")
    @patch("run_tests.click.echo")
    def test_cloud_smoke_handles_missing_report(self, mock_echo, mock_run):
        """Handles case where JSON report doesn't exist."""
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="No tests found")

        with patch.object(Path, "exists", return_value=False):
            exit_code, summary = run_cloud_smoke("azure", "json")

        # Summary should have defaults when no report exists
        assert summary.total == 0
        assert summary.passed == 0
        assert exit_code == 1


class TestRunLoadTest:
    """Tests for run_load_test function.

    This function runs locust load tests and parses the JSON output.
    We test the parsing logic and error handling.
    """

    @patch("run_tests.subprocess.run")
    def test_load_test_parses_locust_output(self, mock_run):
        """Parses locust JSON output correctly."""
        locust_output = json.dumps([
            {
                "name": "/v1/completions",
                "num_requests": 50,
                "num_failures": 2,
            },
            {
                "name": "Aggregated",
                "num_requests": 100,
                "num_failures": 5,
                "current_rps": 3.5,
                "avg_response_time": 250.0,
                "response_times": {"50": 200, "95": 450, "99": 800},
            },
        ])
        mock_run.return_value = MagicMock(returncode=0, stdout=locust_output, stderr="")

        with patch.object(Path, "exists", return_value=False):
            exit_code, metrics = run_load_test(
                base_url="http://localhost:8080",
                duration=60,
                users=10,
                spawn_rate=2,
            )

        assert metrics["duration_seconds"] == 60
        assert metrics["users"] == 10
        assert metrics["requests_total"] == 100
        assert metrics["requests_failed"] == 5
        assert metrics["requests_per_second"] == 3.5
        assert metrics["latency_p50_ms"] == 200
        assert metrics["latency_p95_ms"] == 450
        assert metrics["latency_p99_ms"] == 800

    @patch("run_tests.subprocess.run")
    def test_load_test_handles_json_decode_error(self, mock_run):
        """Handles invalid JSON output gracefully."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="Error: connection refused\nNot JSON",
            stderr=""
        )

        with patch.object(Path, "exists", return_value=False):
            exit_code, metrics = run_load_test(
                base_url="http://localhost:8080",
                duration=30,
                users=5,
                spawn_rate=1,
            )

        # Should have parse error but not crash
        assert "parse_error" in metrics
        assert metrics["duration_seconds"] == 30

    @patch("run_tests.subprocess.run")
    def test_load_test_handles_empty_output(self, mock_run):
        """Handles empty locust output."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        with patch.object(Path, "exists", return_value=False):
            exit_code, metrics = run_load_test(
                base_url="http://localhost:8080",
                duration=10,
                users=1,
                spawn_rate=1,
            )

        # Should have base metrics but no parsed data
        assert metrics["duration_seconds"] == 10
        assert "requests_total" not in metrics

    @patch("run_tests.subprocess.run")
    def test_load_test_loads_custom_metrics_file(self, mock_run):
        """Loads custom metrics file when available."""
        locust_output = json.dumps([{"name": "Aggregated", "num_requests": 10}])
        mock_run.return_value = MagicMock(returncode=0, stdout=locust_output, stderr="")

        custom_metrics = {
            "tokens_generated": 5000,
            "avg_tokens_per_request": 50,
        }

        with patch.object(Path, "exists", return_value=True):
            with patch("builtins.open", MagicMock()):
                with patch("json.load", return_value=custom_metrics):
                    exit_code, metrics = run_load_test(
                        base_url="http://localhost:8080",
                        duration=60,
                        users=10,
                        spawn_rate=2,
                    )

        assert "custom" in metrics
        assert metrics["custom"]["tokens_generated"] == 5000

    @patch("run_tests.subprocess.run")
    def test_load_test_no_aggregated_stats(self, mock_run):
        """Handles locust output without Aggregated entry."""
        locust_output = json.dumps([
            {"name": "/health", "num_requests": 10},
            {"name": "/v1/completions", "num_requests": 20},
        ])
        mock_run.return_value = MagicMock(returncode=0, stdout=locust_output, stderr="")

        with patch.object(Path, "exists", return_value=False):
            exit_code, metrics = run_load_test(
                base_url="http://localhost:8080",
                duration=30,
                users=5,
                spawn_rate=1,
            )

        # Should not have aggregated metrics
        assert "requests_total" not in metrics
        assert metrics["duration_seconds"] == 30


class TestMainCLI:
    """Tests for the main CLI function.

    These tests use click.testing.CliRunner to simulate CLI invocation
    and test the various code paths in main().
    """

    @patch("run_tests.run_cloud_smoke")
    def test_main_cloud_suite_calls_cloud_smoke(self, mock_cloud_smoke):
        """Cloud suite delegates to run_cloud_smoke."""
        from click.testing import CliRunner
        from run_tests import main

        mock_cloud_smoke.return_value = (0, TestSummary(suite="cloud-aws"))

        runner = CliRunner()
        result = runner.invoke(main, ["--suite", "cloud", "--cloud", "aws"])

        mock_cloud_smoke.assert_called_once_with("aws", "text")
        assert result.exit_code == 0

    @patch("run_tests.run_cloud_smoke")
    def test_main_cloud_suite_json_output(self, mock_cloud_smoke):
        """Cloud suite with JSON output."""
        from click.testing import CliRunner
        from run_tests import main

        mock_cloud_smoke.return_value = (0, TestSummary(suite="cloud-gcp"))

        runner = CliRunner()
        result = runner.invoke(main, ["--suite", "cloud", "--cloud", "gcp", "--output", "json"])

        mock_cloud_smoke.assert_called_once_with("gcp", "json")

    @patch("run_tests.check_infrastructure")
    def test_main_unhealthy_infrastructure_exits_2(self, mock_check):
        """Unhealthy infrastructure exits with code 2."""
        from click.testing import CliRunner
        from run_tests import main

        mock_check.return_value = {
            "healthy": False,
            "url": "http://localhost:8080",
            "error": "Connection refused",
            "response_time_ms": None,
        }

        runner = CliRunner()
        result = runner.invoke(main, ["--suite", "smoke", "--url", "http://localhost:8080"])

        assert result.exit_code == 2
        assert "not healthy" in result.output or "Connection refused" in result.output

    @patch("run_tests.check_infrastructure")
    def test_main_unhealthy_infrastructure_json_output(self, mock_check):
        """Unhealthy infrastructure with JSON output."""
        from click.testing import CliRunner
        from run_tests import main

        mock_check.return_value = {
            "healthy": False,
            "url": "http://localhost:8080",
            "error": "Timeout",
            "response_time_ms": None,
        }

        runner = CliRunner()
        result = runner.invoke(main, ["--suite", "smoke", "--output", "json"])

        assert result.exit_code == 2
        # Should output valid JSON
        output_data = json.loads(result.output)
        assert output_data["infrastructure"]["healthy"] is False

    @patch("run_tests.check_infrastructure")
    @patch("run_tests.run_load_test")
    def test_main_load_suite_runs_load_test(self, mock_load_test, mock_check):
        """Load suite runs load test with correct parameters."""
        from click.testing import CliRunner
        from run_tests import main

        mock_check.return_value = {
            "healthy": True,
            "url": "http://localhost:8080",
            "error": None,
            "response_time_ms": 50,
        }
        mock_load_test.return_value = (0, {
            "requests_total": 100,
            "requests_failed": 2,
            "requests_per_second": 3.5,
            "latency_p95_ms": 250,
        })

        runner = CliRunner()
        result = runner.invoke(main, [
            "--suite", "load",
            "--url", "http://localhost:8080",
            "--duration", "30",
            "--users", "5",
            "--spawn-rate", "1",
        ])

        mock_load_test.assert_called_once_with("http://localhost:8080", 30, 5, 1)

    @patch("run_tests.check_infrastructure")
    @patch("run_tests.run_load_test")
    def test_main_load_test_high_error_rate_fails(self, mock_load_test, mock_check):
        """Load test with >5% error rate exits with code 1."""
        from click.testing import CliRunner
        from run_tests import main

        mock_check.return_value = {"healthy": True, "url": "http://localhost:8080", "error": None, "response_time_ms": 50}
        mock_load_test.return_value = (0, {
            "requests_total": 100,
            "requests_failed": 10,  # 10% error rate
        })

        runner = CliRunner()
        result = runner.invoke(main, ["--suite", "load"])

        assert result.exit_code == 1

    @patch("run_tests.check_infrastructure")
    @patch("run_tests.run_load_test")
    def test_main_load_test_low_error_rate_passes(self, mock_load_test, mock_check):
        """Load test with <5% error rate passes."""
        from click.testing import CliRunner
        from run_tests import main

        mock_check.return_value = {"healthy": True, "url": "http://localhost:8080", "error": None, "response_time_ms": 50}
        mock_load_test.return_value = (0, {
            "requests_total": 100,
            "requests_failed": 2,  # 2% error rate
        })

        runner = CliRunner()
        result = runner.invoke(main, ["--suite", "load"])

        assert result.exit_code == 0

    @patch("run_tests.check_infrastructure")
    @patch("run_tests.run_pytest")
    def test_main_smoke_suite_runs_pytest(self, mock_pytest, mock_check):
        """Smoke suite runs pytest with infrastructure info."""
        from click.testing import CliRunner
        from run_tests import main

        mock_check.return_value = {"healthy": True, "url": "http://localhost:8080", "error": None, "response_time_ms": 50}
        mock_pytest.return_value = (0, TestSummary(suite="smoke", total=5, passed=5))

        runner = CliRunner()
        result = runner.invoke(main, ["--suite", "smoke"])

        mock_pytest.assert_called_once()
        assert result.exit_code == 0

    @patch("run_tests.check_infrastructure")
    @patch("run_tests.run_pytest")
    def test_main_text_output_shows_failures(self, mock_pytest, mock_check):
        """Text output displays failed tests with truncated errors."""
        from click.testing import CliRunner
        from run_tests import main

        mock_check.return_value = {"healthy": True, "url": "http://localhost:8080", "error": None, "response_time_ms": 50}

        failed_test = TestResult(
            name="test_failure",
            status="failed",
            duration_seconds=1.0,
            error="AssertionError: " + "x" * 300,  # Long error message
        )
        mock_pytest.return_value = (1, TestSummary(
            suite="smoke",
            total=2,
            passed=1,
            failed=1,
            tests=[failed_test],
        ))

        runner = CliRunner()
        result = runner.invoke(main, ["--suite", "smoke", "--output", "text"])

        assert "Failed tests:" in result.output
        assert "test_failure" in result.output
        # Error should be truncated to ~200 chars
        assert "..." in result.output

    @patch("run_tests.check_infrastructure")
    @patch("run_tests.run_load_test")
    def test_main_load_test_text_output_shows_metrics(self, mock_load_test, mock_check):
        """Load test text output shows RPS and latency."""
        from click.testing import CliRunner
        from run_tests import main

        mock_check.return_value = {"healthy": True, "url": "http://localhost:8080", "error": None, "response_time_ms": 50}
        mock_load_test.return_value = (0, {
            "requests_total": 100,
            "requests_failed": 1,
            "requests_per_second": 5.5,
            "latency_p95_ms": 350,
        })

        runner = CliRunner()
        result = runner.invoke(main, ["--suite", "load", "--output", "text"])

        assert "Load Test Results:" in result.output
        assert "RPS:" in result.output
        assert "P95 Latency:" in result.output

    @patch("run_tests.check_infrastructure")
    @patch("run_tests.run_pytest")
    def test_main_json_output_is_valid(self, mock_pytest, mock_check):
        """JSON output is valid JSON."""
        from click.testing import CliRunner
        from run_tests import main

        mock_check.return_value = {"healthy": True, "url": "http://localhost:8080", "error": None, "response_time_ms": 50}
        mock_pytest.return_value = (0, TestSummary(suite="smoke", total=3, passed=3))

        runner = CliRunner()
        result = runner.invoke(main, ["--suite", "smoke", "--output", "json"])

        # Should be valid JSON
        output_data = json.loads(result.output)
        assert output_data["suite"] == "smoke"
        assert output_data["total"] == 3
        assert output_data["passed"] == 3

    @patch("run_tests.check_infrastructure")
    @patch("run_tests.run_pytest")
    def test_main_integration_suite(self, mock_pytest, mock_check):
        """Integration suite uses correct marker."""
        from click.testing import CliRunner
        from run_tests import main

        mock_check.return_value = {"healthy": True, "url": "http://localhost:8080", "error": None, "response_time_ms": 50}
        mock_pytest.return_value = (0, TestSummary(suite="integration", total=10, passed=10))

        runner = CliRunner()
        result = runner.invoke(main, ["--suite", "integration"])

        # run_pytest should be called with "integration" suite
        assert mock_pytest.call_args[0][0] == "integration"


class TestRunPytestSuitePaths:
    """Tests for run_pytest suite path selection logic."""

    @patch("run_tests.subprocess.run")
    def test_integration_suite_uses_integration_file(self, mock_run):
        """Integration suite runs test_integration.py with marker."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        with patch.object(Path, "exists", return_value=False):
            exit_code, summary = run_pytest("integration", "http://localhost:8080")

        call_args = mock_run.call_args[0][0]
        assert "test_integration.py" in str(call_args)
        assert "-m" in call_args
        assert "integration" in call_args

    @patch("run_tests.subprocess.run")
    def test_custom_suite_uses_custom_file(self, mock_run):
        """Custom suite name creates path to test_{name}.py."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        with patch.object(Path, "exists", return_value=False):
            # Using a custom suite name "custom"
            exit_code, summary = run_pytest("custom", "http://localhost:8080")

        call_args = mock_run.call_args[0][0]
        # Should use test_custom.py
        assert "test_custom.py" in str(call_args)

    @patch("run_tests.subprocess.run")
    def test_extra_args_passed_to_pytest(self, mock_run):
        """Extra arguments are appended to pytest command."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        with patch.object(Path, "exists", return_value=False):
            exit_code, summary = run_pytest(
                "smoke",
                "http://localhost:8080",
                extra_args=["--capture=no", "-x"]
            )

        call_args = mock_run.call_args[0][0]
        assert "--capture=no" in call_args
        assert "-x" in call_args

    @patch("run_tests.subprocess.run")
    def test_pytest_env_includes_inference_url(self, mock_run):
        """INFERENCE_URL environment variable is set."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        with patch.object(Path, "exists", return_value=False):
            exit_code, summary = run_pytest("smoke", "http://custom:9000")

        # Check environment
        env = mock_run.call_args[1]["env"]
        assert env["INFERENCE_URL"] == "http://custom:9000"


class TestCloudSmokeTextOutput:
    """Tests for cloud smoke text output formatting (lines 221-227)."""

    @patch("run_tests.subprocess.run")
    @patch("run_tests.click.echo")
    def test_text_output_no_failures_skips_failure_section(self, mock_echo, mock_run):
        """Text output with no failures doesn't show failure section."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        mock_report = {
            "summary": {"total": 3, "passed": 3, "failed": 0},
            "tests": [
                {"nodeid": "test_aws::test_a", "outcome": "passed", "duration": 1.0},
                {"nodeid": "test_aws::test_b", "outcome": "passed", "duration": 2.0},
                {"nodeid": "test_aws::test_c", "outcome": "passed", "duration": 3.0},
            ],
        }

        with patch.object(Path, "exists", return_value=True):
            with patch("builtins.open", MagicMock()):
                with patch("json.load", return_value=mock_report):
                    exit_code, summary = run_cloud_smoke("aws", "text")

        # Check that "Failed tests:" was NOT printed
        echo_calls = [str(call) for call in mock_echo.call_args_list]
        assert not any("Failed tests" in str(call) for call in echo_calls)
        assert summary.failed == 0

    @patch("run_tests.subprocess.run")
    @patch("run_tests.click.echo")
    def test_text_output_with_failures_shows_failure_section(self, mock_echo, mock_run):
        """Text output with failures shows failure section."""
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")

        mock_report = {
            "summary": {"total": 2, "passed": 1, "failed": 1},
            "tests": [
                {"nodeid": "test_gcp::test_pass", "outcome": "passed", "duration": 1.0},
                {"nodeid": "test_gcp::test_fail", "outcome": "failed", "duration": 2.0,
                 "call": {"longrepr": "AssertionError"}},
            ],
        }

        with patch.object(Path, "exists", return_value=True):
            with patch("builtins.open", MagicMock()):
                with patch("json.load", return_value=mock_report):
                    exit_code, summary = run_cloud_smoke("gcp", "text")

        # Check that failed test was printed
        echo_calls = [str(call) for call in mock_echo.call_args_list]
        assert any("Failed tests" in str(call) for call in echo_calls)


class TestLocustOutputParsing:
    """Tests for locust JSON output parsing edge cases (lines 273-288)."""

    @patch("run_tests.subprocess.run")
    def test_locust_output_with_empty_list(self, mock_run):
        """Handles locust output that is an empty list."""
        mock_run.return_value = MagicMock(returncode=0, stdout="[]", stderr="")

        with patch.object(Path, "exists", return_value=False):
            exit_code, metrics = run_load_test(
                base_url="http://localhost:8080",
                duration=30,
                users=5,
                spawn_rate=1,
            )

        # Should not have aggregated metrics
        assert "requests_total" not in metrics
        assert metrics["duration_seconds"] == 30

    @patch("run_tests.subprocess.run")
    def test_locust_output_partial_aggregated(self, mock_run):
        """Handles aggregated entry with partial fields."""
        locust_output = json.dumps([
            {
                "name": "Aggregated",
                "num_requests": 50,
                # Missing other fields
            },
        ])
        mock_run.return_value = MagicMock(returncode=0, stdout=locust_output, stderr="")

        with patch.object(Path, "exists", return_value=False):
            exit_code, metrics = run_load_test(
                base_url="http://localhost:8080",
                duration=30,
                users=5,
                spawn_rate=1,
            )

        assert metrics["requests_total"] == 50
        assert metrics["requests_failed"] == 0  # Default value
        assert metrics["requests_per_second"] == 0  # Default value


class TestTextOutputErrorTruncation:
    """Tests for text output error truncation (lines 395-398)."""

    @patch("run_tests.check_infrastructure")
    @patch("run_tests.run_pytest")
    def test_text_output_no_error_message(self, mock_pytest, mock_check):
        """Text output handles test failures without error message."""
        from click.testing import CliRunner
        from run_tests import main

        mock_check.return_value = {"healthy": True, "url": "http://localhost:8080", "error": None, "response_time_ms": 50}

        # Failed test without error message
        failed_test = TestResult(
            name="test_no_error",
            status="failed",
            duration_seconds=1.0,
            error=None,  # No error message
        )
        mock_pytest.return_value = (1, TestSummary(
            suite="smoke",
            total=1,
            passed=0,
            failed=1,
            tests=[failed_test],
        ))

        runner = CliRunner()
        result = runner.invoke(main, ["--suite", "smoke", "--output", "text"])

        assert "Failed tests:" in result.output
        assert "test_no_error" in result.output
        # Should not crash without error message

    @patch("run_tests.check_infrastructure")
    @patch("run_tests.run_pytest")
    def test_text_output_short_error_message(self, mock_pytest, mock_check):
        """Text output handles short error messages without truncation."""
        from click.testing import CliRunner
        from run_tests import main

        mock_check.return_value = {"healthy": True, "url": "http://localhost:8080", "error": None, "response_time_ms": 50}

        failed_test = TestResult(
            name="test_short_error",
            status="failed",
            duration_seconds=1.0,
            error="Short error",  # Short error, no truncation needed
        )
        mock_pytest.return_value = (1, TestSummary(
            suite="smoke",
            total=1,
            passed=0,
            failed=1,
            tests=[failed_test],
        ))

        runner = CliRunner()
        result = runner.invoke(main, ["--suite", "smoke", "--output", "text"])

        assert "Short error" in result.output


class TestLoadTestTextOutput:
    """Tests for load test text output edge cases."""

    @patch("run_tests.check_infrastructure")
    @patch("run_tests.run_load_test")
    def test_load_test_text_no_metrics(self, mock_load_test, mock_check):
        """Load test text output handles missing metrics gracefully."""
        from click.testing import CliRunner
        from run_tests import main

        mock_check.return_value = {"healthy": True, "url": "http://localhost:8080", "error": None, "response_time_ms": 50}
        # Missing requests_per_second and latency_p95_ms
        mock_load_test.return_value = (0, {
            "requests_total": 100,
            "requests_failed": 1,
        })

        runner = CliRunner()
        result = runner.invoke(main, ["--suite", "load", "--output", "text"])

        assert "Load Test Results:" in result.output
        assert "N/A" in result.output  # Missing values shown as N/A
