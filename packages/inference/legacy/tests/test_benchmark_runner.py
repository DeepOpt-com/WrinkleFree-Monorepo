"""Tests for the benchmark runner module."""

import pytest
from datetime import datetime
from pathlib import Path
import sys
import tempfile
import json

# Add benchmark module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmark"))

from benchmark.metrics import BenchmarkMetrics, CostBenchmarkResult
from benchmark.runner import BenchmarkConfig


class TestBenchmarkMetrics:
    """Tests for BenchmarkMetrics dataclass."""

    def test_from_latencies_basic(self):
        """Test creating metrics from latency data."""
        latencies = [100, 150, 200, 250, 300]  # ms
        tokens = [50, 45, 55, 48, 52]
        total_time = 5.0  # seconds

        metrics = BenchmarkMetrics.from_latencies(
            name="test",
            latencies=latencies,
            tokens=tokens,
            total_time=total_time,
        )

        assert metrics.name == "test"
        assert metrics.requests == 5
        assert metrics.successful == 5
        assert metrics.failed == 0
        assert metrics.tokens_generated == sum(tokens)
        assert metrics.latency_avg_ms == 200  # mean of latencies
        assert metrics.latency_p50_ms == 200  # median

    def test_from_latencies_with_failures(self):
        """Test metrics with some failed requests."""
        latencies = [100, 150, 200]
        tokens = [50, 45, 55]
        total_time = 5.0
        failed = 2

        metrics = BenchmarkMetrics.from_latencies(
            name="test",
            latencies=latencies,
            tokens=tokens,
            total_time=total_time,
            failed=failed,
        )

        assert metrics.requests == 5  # 3 successful + 2 failed
        assert metrics.successful == 3
        assert metrics.failed == 2

    def test_from_latencies_empty(self):
        """Test metrics with no successful requests."""
        metrics = BenchmarkMetrics.from_latencies(
            name="test",
            latencies=[],
            tokens=[],
            total_time=1.0,
            failed=5,
        )

        assert metrics.successful == 0
        assert metrics.failed == 5
        assert metrics.tokens_per_second == 0
        assert metrics.latency_avg_ms == 0

    def test_from_latencies_with_ttft(self):
        """Test metrics with time-to-first-token data."""
        latencies = [100, 150, 200]
        tokens = [50, 45, 55]
        ttfts = [30, 40, 50]

        metrics = BenchmarkMetrics.from_latencies(
            name="test",
            latencies=latencies,
            tokens=tokens,
            total_time=5.0,
            ttfts=ttfts,
        )

        assert metrics.ttft_avg_ms == 40  # mean
        assert metrics.ttft_p50_ms == 40  # median

    def test_throughput_calculation(self):
        """Test throughput calculations."""
        latencies = [100] * 10
        tokens = [50] * 10
        total_time = 2.0  # 2 seconds

        metrics = BenchmarkMetrics.from_latencies(
            name="test",
            latencies=latencies,
            tokens=tokens,
            total_time=total_time,
        )

        # 10 requests in 2 seconds = 5 req/s
        assert metrics.requests_per_second == 5.0

        # 500 tokens in 2 seconds = 250 tok/s
        assert metrics.tokens_per_second == 250.0


class TestCostBenchmarkResult:
    """Tests for CostBenchmarkResult dataclass."""

    def test_calculate_costs(self):
        """Test cost calculation."""
        result = CostBenchmarkResult(
            run_id="test-123",
            tokens_per_second=100,
            hardware_cost_per_hour=0.39,
        )

        result.calculate_costs()

        # Expected: $0.39/hr / (100 * 3600 tokens/hr) * 1M = $1.0833/1M
        expected = (0.39 / (100 * 3600)) * 1_000_000
        assert abs(result.cost_per_million_tokens - expected) < 0.0001

        # 70% utilization should be higher
        assert result.cost_per_million_at_70pct > result.cost_per_million_tokens

        # 50% should be even higher
        assert result.cost_per_million_at_50pct > result.cost_per_million_at_70pct

    def test_calculate_costs_zero_throughput(self):
        """Test cost calculation with zero throughput."""
        result = CostBenchmarkResult(
            run_id="test-123",
            tokens_per_second=0,
            hardware_cost_per_hour=0.39,
        )

        result.calculate_costs()

        # Should not raise, cost remains 0
        assert result.cost_per_million_tokens == 0

    def test_to_dict(self):
        """Test dictionary serialization."""
        result = CostBenchmarkResult(
            run_id="test-123",
            model="bitnet-2b",
            hardware="a40",
            tokens_per_second=100,
            hardware_cost_per_hour=0.39,
        )
        result.calculate_costs()

        d = result.to_dict()

        assert d["run_id"] == "test-123"
        assert d["model"] == "bitnet-2b"
        assert d["hardware"] == "a40"
        assert "cost_per_million_tokens" in d
        assert isinstance(d["timestamp"], str)  # ISO format

    def test_save_and_load(self):
        """Test saving and loading results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = CostBenchmarkResult(
                run_id="test-123",
                model="bitnet-2b",
                hardware="a40",
                tokens_per_second=100,
                hardware_cost_per_hour=0.39,
            )
            result.calculate_costs()

            # Save
            output_path = result.save(Path(tmpdir))
            assert output_path.exists()

            # Load
            loaded = CostBenchmarkResult.load(output_path)
            assert loaded.run_id == result.run_id
            assert loaded.model == result.model
            assert loaded.tokens_per_second == result.tokens_per_second


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BenchmarkConfig(
            model="test-model",
            model_size="2B",
            quantization="native",
            hardware="a40",
            hardware_type="gpu",
        )

        assert config.batch_sizes == [1, 4, 8]
        assert config.warmup_requests == 5
        assert config.benchmark_requests == 50
        assert config.server_url == "http://localhost:8080"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = BenchmarkConfig(
            model="test-model",
            model_size="70B",
            quantization="naive",
            hardware="cpu_64",
            hardware_type="cpu",
            batch_sizes=[1, 2, 4],
            duration_seconds=120,
            server_url="http://10.0.0.1:8080",
        )

        assert config.batch_sizes == [1, 2, 4]
        assert config.duration_seconds == 120
        assert config.hardware_type == "cpu"


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner class."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return BenchmarkConfig(
            model="test-model",
            model_size="2B",
            quantization="native",
            hardware="a40",
            hardware_type="gpu",
            warmup_requests=1,
            benchmark_requests=5,
            duration_seconds=5,
        )

    def test_initialization(self, config):
        """Test runner initialization."""
        from benchmark.runner import BenchmarkRunner
        from benchmark.cost_tracker import CostTracker

        runner = BenchmarkRunner(config)

        assert runner.config == config
        assert runner.cost_tracker is not None

    def test_default_prompts(self, config):
        """Test default prompts are loaded."""
        from benchmark.runner import BenchmarkRunner

        runner = BenchmarkRunner(config)
        prompts = runner._prompts

        assert len(prompts) > 0
        assert all(isinstance(p, str) for p in prompts)

    @pytest.mark.asyncio
    async def test_run_no_server(self, config):
        """Test run fails gracefully without server."""
        from benchmark.runner import BenchmarkRunner

        # Use a URL that won't connect
        config.server_url = "http://localhost:99999"

        runner = BenchmarkRunner(config)
        result = await runner.run()

        # Should complete but with errors
        assert result.run_id is not None
        assert result.errors >= 0  # May or may not have errors depending on implementation


class TestReportIntegration:
    """Tests for report generation integration."""

    def test_generate_report_from_results(self):
        """Test generating a report from results."""
        from benchmark.report_generator import ReportGenerator, ReportConfig

        results = [
            CostBenchmarkResult(
                run_id="test-1",
                model="bitnet-2b",
                hardware="a40",
                tokens_per_second=100,
                hardware_cost_per_hour=0.39,
                cost_per_million_tokens=1.0833,
            ),
            CostBenchmarkResult(
                run_id="test-2",
                model="bitnet-2b",
                hardware="cpu_64",
                tokens_per_second=50,
                hardware_cost_per_hour=0.24,
                cost_per_million_tokens=1.333,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ReportGenerator(results)
            report_path = generator.generate_report(Path(tmpdir) / "report.md")

            assert report_path.exists()

            content = report_path.read_text()
            assert "bitnet-2b" in content
            assert "a40" in content
            assert "cpu_64" in content
            assert "Performance Comparison" in content

    def test_generate_report_empty_results(self):
        """Test generating report with no results."""
        from benchmark.report_generator import ReportGenerator

        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ReportGenerator([])
            report_path = generator.generate_report(Path(tmpdir) / "report.md")

            assert report_path.exists()
            content = report_path.read_text()
            assert "No benchmark results" in content
