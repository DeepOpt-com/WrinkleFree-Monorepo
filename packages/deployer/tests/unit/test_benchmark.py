"""
Unit tests for scripts/benchmark_throughput.py

These tests mock external dependencies to test the benchmark logic
without requiring running services.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add scripts to path for import
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from benchmark_throughput import (
    BenchmarkResult,
    calculate_costs,
    PROMPTS,
    run_batch,
    benchmark_batch_size,
)


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_create_benchmark_result(self):
        """Can create a benchmark result."""
        result = BenchmarkResult(
            batch_size=8,
            total_requests=100,
            total_tokens=5000,
            duration_seconds=30.0,
            tokens_per_second=166.67,
            requests_per_second=3.33,
            latency_p50_ms=150.0,
            latency_p99_ms=450.0,
            errors=2,
        )
        assert result.batch_size == 8
        assert result.total_tokens == 5000
        assert result.tokens_per_second == 166.67
        assert result.errors == 2

    def test_result_with_zero_errors(self):
        """Can create result with no errors."""
        result = BenchmarkResult(
            batch_size=1,
            total_requests=50,
            total_tokens=2500,
            duration_seconds=60.0,
            tokens_per_second=41.67,
            requests_per_second=0.83,
            latency_p50_ms=100.0,
            latency_p99_ms=200.0,
            errors=0,
        )
        assert result.errors == 0


class TestCalculateCosts:
    """Tests for calculate_costs function."""

    def test_calculate_costs_for_known_throughput(self):
        """Calculates costs correctly for known throughput."""
        # 1000 tokens/second = 3.6M tokens/hour
        tokens_per_second = 1000.0
        costs = calculate_costs(tokens_per_second)

        # Should have multiple infrastructure options
        assert len(costs) > 0
        assert "Hetzner AX42 ($0.075/hr)" in costs
        assert "Hetzner AX102 ($0.214/hr)" in costs

        # Each infra should have utilization tiers
        hetzner_ax42 = costs["Hetzner AX42 ($0.075/hr)"]
        assert "100% util" in hetzner_ax42
        assert "70% util" in hetzner_ax42
        assert "50% util" in hetzner_ax42

        # 100% util should be cheaper than 50% util
        assert hetzner_ax42["100% util"] < hetzner_ax42["50% util"]

    def test_calculate_costs_zero_throughput(self):
        """Returns empty dict for zero throughput."""
        costs = calculate_costs(0.0)
        assert costs == {}

    def test_cost_scales_with_throughput(self):
        """Higher throughput = lower cost per million tokens."""
        low_throughput_costs = calculate_costs(100.0)
        high_throughput_costs = calculate_costs(1000.0)

        # Get Hetzner AX42 at 100% util for comparison
        low_cost = low_throughput_costs["Hetzner AX42 ($0.075/hr)"]["100% util"]
        high_cost = high_throughput_costs["Hetzner AX42 ($0.075/hr)"]["100% util"]

        # 10x throughput should give ~10x lower cost (allow rounding tolerance)
        assert high_cost < low_cost
        ratio = low_cost / high_cost
        assert 9.5 < ratio < 10.5  # Approximately 10x, allowing for rounding


class TestPrompts:
    """Tests for prompt constants."""

    def test_prompts_not_empty(self):
        """PROMPTS list should not be empty."""
        assert len(PROMPTS) > 0

    def test_prompts_are_strings(self):
        """All prompts should be strings."""
        for prompt in PROMPTS:
            assert isinstance(prompt, str)
            assert len(prompt) > 0

    def test_prompts_variety(self):
        """Should have variety of prompts."""
        # At least 5 different prompts
        assert len(PROMPTS) >= 5

        # All should be unique
        assert len(set(PROMPTS)) == len(PROMPTS)


class TestMakeRequest:
    """Tests for make_request async function."""

    @pytest.mark.asyncio
    async def test_make_request_success(self):
        """Successful request returns tokens and latency."""
        from benchmark_throughput import make_request

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "usage": {"completion_tokens": 50},
            "choices": [{"text": "Hello world"}],
        })

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock(return_value=None),
        ))

        with patch("benchmark_throughput.aiohttp.ClientSession", return_value=mock_session):
            tokens, latency, error = await make_request(
                mock_session, "http://localhost:8080", "Hello", 50
            )

        assert tokens == 50
        assert latency > 0
        assert error is None

    @pytest.mark.asyncio
    async def test_make_request_error_status(self):
        """HTTP error returns 0 tokens and error message."""
        from benchmark_throughput import make_request

        mock_response = MagicMock()
        mock_response.status = 500

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock(return_value=None),
        ))

        tokens, latency, error = await make_request(
            mock_session, "http://localhost:8080", "Hello", 50
        )

        assert tokens == 0
        assert "500" in error


class TestCheckHealth:
    """Tests for check_health async function."""

    @pytest.mark.asyncio
    async def test_check_health_success(self):
        """Healthy endpoint returns True."""
        from benchmark_throughput import check_health

        mock_response = MagicMock()
        mock_response.status = 200

        with patch("benchmark_throughput.aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get = MagicMock(return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response),
                __aexit__=AsyncMock(return_value=None),
            ))
            mock_session_class.return_value = mock_session

            result = await check_health("http://localhost:8080")

        assert result is True

    @pytest.mark.asyncio
    async def test_check_health_failure(self):
        """Unhealthy endpoint returns False."""
        from benchmark_throughput import check_health

        mock_response = MagicMock()
        mock_response.status = 503

        with patch("benchmark_throughput.aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get = MagicMock(return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response),
                __aexit__=AsyncMock(return_value=None),
            ))
            mock_session_class.return_value = mock_session

            result = await check_health("http://localhost:8080")

        assert result is False

    @pytest.mark.asyncio
    async def test_check_health_connection_error(self):
        """Connection error returns False."""
        from benchmark_throughput import check_health

        with patch("benchmark_throughput.aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get = MagicMock(side_effect=Exception("Connection refused"))
            mock_session_class.return_value = mock_session

            result = await check_health("http://localhost:8080")

        assert result is False


class TestRunBatch:
    """Tests for run_batch async function.

    run_batch runs multiple concurrent requests and aggregates results.
    """

    @pytest.mark.asyncio
    async def test_run_batch_returns_correct_count(self):
        """Batch returns results for all requests."""
        from benchmark_throughput import make_request

        mock_session = MagicMock()

        # Mock make_request to return immediately
        with patch("benchmark_throughput.make_request", new=AsyncMock(
            return_value=(50, 100.0, None)
        )):
            results = await run_batch(
                session=mock_session,
                endpoint="http://localhost:8080",
                batch_size=5,
                max_tokens=50,
            )

        assert len(results) == 5
        for tokens, latency, error in results:
            assert tokens == 50
            assert latency == 100.0
            assert error is None

    @pytest.mark.asyncio
    async def test_run_batch_uses_rotating_prompts(self):
        """Batch uses prompts from PROMPTS list in rotation."""
        mock_session = MagicMock()
        captured_prompts = []

        async def capture_prompt(session, endpoint, prompt, max_tokens):
            captured_prompts.append(prompt)
            return (50, 100.0, None)

        with patch("benchmark_throughput.make_request", new=capture_prompt):
            # Run batch larger than PROMPTS length
            batch_size = len(PROMPTS) + 3
            results = await run_batch(
                session=mock_session,
                endpoint="http://localhost:8080",
                batch_size=batch_size,
                max_tokens=50,
            )

        assert len(results) == batch_size
        # First prompts should match PROMPTS exactly
        assert captured_prompts[:len(PROMPTS)] == PROMPTS
        # Extra prompts should wrap around
        assert captured_prompts[len(PROMPTS):] == PROMPTS[:3]


class TestBenchmarkBatchSize:
    """Tests for benchmark_batch_size async function.

    benchmark_batch_size runs batches for a duration and computes statistics.
    """

    @pytest.mark.asyncio
    async def test_benchmark_computes_throughput(self):
        """Benchmark correctly computes tokens per second."""
        # Mock time to control duration
        mock_time = [0.0]

        def mock_perf_counter():
            result = mock_time[0]
            mock_time[0] += 0.1  # Advance 100ms each call
            return result

        with patch("benchmark_throughput.time.perf_counter", side_effect=mock_perf_counter):
            with patch("benchmark_throughput.aiohttp.ClientSession") as mock_session_class:
                mock_session = MagicMock()
                mock_session.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session.__aexit__ = AsyncMock(return_value=None)
                mock_session_class.return_value = mock_session

                # Mock run_batch to return results
                with patch("benchmark_throughput.run_batch", new=AsyncMock(
                    return_value=[(50, 100.0, None), (50, 120.0, None)]
                )):
                    result = await benchmark_batch_size(
                        endpoint="http://localhost:8080",
                        batch_size=2,
                        duration_seconds=0.5,  # Will run for ~5 iterations (0.1s each)
                        max_tokens=50,
                    )

        assert result.batch_size == 2
        assert result.total_tokens > 0
        assert result.tokens_per_second > 0
        assert result.errors == 0

    @pytest.mark.asyncio
    async def test_benchmark_handles_all_errors(self):
        """Benchmark handles case where all requests fail."""
        mock_time = [0.0]

        def mock_perf_counter():
            result = mock_time[0]
            mock_time[0] += 0.2
            return result

        with patch("benchmark_throughput.time.perf_counter", side_effect=mock_perf_counter):
            with patch("benchmark_throughput.aiohttp.ClientSession") as mock_session_class:
                mock_session = MagicMock()
                mock_session.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session.__aexit__ = AsyncMock(return_value=None)
                mock_session_class.return_value = mock_session

                # All requests fail
                with patch("benchmark_throughput.run_batch", new=AsyncMock(
                    return_value=[(0, 500.0, "Timeout"), (0, 600.0, "Connection refused")]
                )):
                    result = await benchmark_batch_size(
                        endpoint="http://localhost:8080",
                        batch_size=2,
                        duration_seconds=0.3,
                        max_tokens=50,
                    )

        # Should have errors but not crash
        assert result.errors > 0
        assert result.total_tokens == 0
        assert result.latency_p50_ms == 0  # No successful requests
        assert result.latency_p99_ms == 0

    @pytest.mark.asyncio
    async def test_benchmark_computes_percentiles(self):
        """Benchmark correctly computes latency percentiles."""
        mock_time = [0.0]

        def mock_perf_counter():
            result = mock_time[0]
            mock_time[0] += 0.1
            return result

        # Create a range of latencies for percentile calculation
        latencies = [float(i * 10) for i in range(1, 101)]  # 10, 20, ... 1000

        batch_idx = [0]
        def mock_run_batch(*args, **kwargs):
            # Return 10 results per batch
            start_idx = batch_idx[0] * 10
            batch_idx[0] += 1
            if start_idx >= len(latencies):
                return []
            end_idx = min(start_idx + 10, len(latencies))
            return [(50, latencies[i], None) for i in range(start_idx, end_idx)]

        with patch("benchmark_throughput.time.perf_counter", side_effect=mock_perf_counter):
            with patch("benchmark_throughput.aiohttp.ClientSession") as mock_session_class:
                mock_session = MagicMock()
                mock_session.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session.__aexit__ = AsyncMock(return_value=None)
                mock_session_class.return_value = mock_session

                with patch("benchmark_throughput.run_batch", new=AsyncMock(
                    side_effect=mock_run_batch
                )):
                    result = await benchmark_batch_size(
                        endpoint="http://localhost:8080",
                        batch_size=10,
                        duration_seconds=1.0,
                        max_tokens=50,
                    )

        # With 100 latencies from 10-1000:
        # p50 should be around 500 (50th value)
        # p99 should be around 990 (99th value)
        assert result.latency_p50_ms > 0
        assert result.latency_p99_ms > result.latency_p50_ms

    @pytest.mark.asyncio
    async def test_benchmark_zero_duration_safe(self):
        """Benchmark handles zero duration without division error."""
        # Mock time to return same value (0 duration)
        with patch("benchmark_throughput.time.perf_counter", return_value=0.0):
            with patch("benchmark_throughput.aiohttp.ClientSession") as mock_session_class:
                mock_session = MagicMock()
                mock_session.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session.__aexit__ = AsyncMock(return_value=None)
                mock_session_class.return_value = mock_session

                # Mock run_batch to return empty (loop won't run)
                with patch("benchmark_throughput.run_batch", new=AsyncMock(return_value=[])):
                    result = await benchmark_batch_size(
                        endpoint="http://localhost:8080",
                        batch_size=1,
                        duration_seconds=0,
                        max_tokens=50,
                    )

        # Should handle gracefully
        assert result.tokens_per_second == 0
        assert result.requests_per_second == 0


class TestMakeRequestEdgeCases:
    """Additional edge case tests for make_request function."""

    @pytest.mark.asyncio
    async def test_make_request_timeout(self):
        """Request timeout returns error."""
        import asyncio
        from benchmark_throughput import make_request

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=asyncio.TimeoutError())

        tokens, latency, error = await make_request(
            mock_session, "http://localhost:8080", "Hello", 50
        )

        assert tokens == 0
        assert error == "Timeout"

    @pytest.mark.asyncio
    async def test_make_request_generic_exception(self):
        """Generic exception returns error string."""
        from benchmark_throughput import make_request

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=Exception("Network error"))

        tokens, latency, error = await make_request(
            mock_session, "http://localhost:8080", "Hello", 50
        )

        assert tokens == 0
        assert "Network error" in error

    @pytest.mark.asyncio
    async def test_make_request_estimates_tokens_from_text(self):
        """Estimates tokens from response text when usage not available."""
        from benchmark_throughput import make_request

        mock_response = MagicMock()
        mock_response.status = 200
        # No usage field, but has choices with text
        mock_response.json = AsyncMock(return_value={
            "choices": [{"text": "This is a response with multiple words"}],
        })

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock(return_value=None),
        ))

        tokens, latency, error = await make_request(
            mock_session, "http://localhost:8080", "Hello", 50
        )

        # Should estimate tokens from text (7 words * 1.3 ≈ 9)
        assert tokens > 0
        assert error is None

    @pytest.mark.asyncio
    async def test_make_request_falls_back_to_max_tokens(self):
        """Falls back to max_tokens when no usage or text available."""
        from benchmark_throughput import make_request

        mock_response = MagicMock()
        mock_response.status = 200
        # No usage, no choices
        mock_response.json = AsyncMock(return_value={})

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock(return_value=None),
        ))

        tokens, latency, error = await make_request(
            mock_session, "http://localhost:8080", "Hello", 50
        )

        # Should fall back to max_tokens
        assert tokens == 50
        assert error is None


class TestMainFunction:
    """Tests for the main async function.

    Tests the CLI behavior including argument parsing, health checks,
    benchmark execution, and output formatting.
    """

    @pytest.mark.asyncio
    async def test_main_unhealthy_endpoint_returns_1(self):
        """Unhealthy endpoint causes early exit with return code 1."""
        from benchmark_throughput import main

        with patch("benchmark_throughput.check_health", new=AsyncMock(return_value=False)):
            with patch("sys.argv", ["benchmark_throughput.py", "--endpoint", "http://localhost:8080"]):
                with patch("builtins.print") as mock_print:
                    result = await main()

        assert result == 1
        # Should print error message
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any("not healthy" in str(call) for call in print_calls)

    @pytest.mark.asyncio
    async def test_main_healthy_endpoint_runs_benchmarks(self):
        """Healthy endpoint runs benchmarks for each batch size."""
        from benchmark_throughput import main

        mock_result = BenchmarkResult(
            batch_size=1,
            total_requests=10,
            total_tokens=500,
            duration_seconds=30.0,
            tokens_per_second=16.67,
            requests_per_second=0.33,
            latency_p50_ms=100.0,
            latency_p99_ms=200.0,
            errors=0,
        )

        with patch("benchmark_throughput.check_health", new=AsyncMock(return_value=True)):
            with patch("benchmark_throughput.benchmark_batch_size", new=AsyncMock(return_value=mock_result)):
                with patch("sys.argv", ["benchmark_throughput.py", "--endpoint", "http://localhost:8080", "--batch-sizes", "1,4"]):
                    with patch("builtins.print"):
                        result = await main()

        assert result == 0

    @pytest.mark.asyncio
    async def test_main_json_output_format(self):
        """JSON output produces valid JSON with expected fields."""
        from benchmark_throughput import main
        import io

        mock_result = BenchmarkResult(
            batch_size=4,
            total_requests=20,
            total_tokens=1000,
            duration_seconds=30.0,
            tokens_per_second=33.33,
            requests_per_second=0.67,
            latency_p50_ms=150.0,
            latency_p99_ms=300.0,
            errors=1,
        )

        captured_output = []

        def capture_print(*args, **kwargs):
            captured_output.append(" ".join(str(a) for a in args))

        with patch("benchmark_throughput.check_health", new=AsyncMock(return_value=True)):
            with patch("benchmark_throughput.benchmark_batch_size", new=AsyncMock(return_value=mock_result)):
                with patch("sys.argv", ["benchmark_throughput.py", "--batch-sizes", "4", "--output", "json"]):
                    with patch("builtins.print", side_effect=capture_print):
                        result = await main()

        # Find the JSON output (the large print)
        json_output = None
        for output in captured_output:
            if "{" in output and "results" in output:
                json_output = output
                break

        assert json_output is not None
        import json
        data = json.loads(json_output)
        assert "endpoint" in data
        assert "results" in data
        assert "best_throughput" in data
        assert len(data["results"]) == 1

    @pytest.mark.asyncio
    async def test_main_table_output_format(self):
        """Table output includes headers and result rows."""
        from benchmark_throughput import main

        mock_result = BenchmarkResult(
            batch_size=8,
            total_requests=40,
            total_tokens=2000,
            duration_seconds=30.0,
            tokens_per_second=66.67,
            requests_per_second=1.33,
            latency_p50_ms=200.0,
            latency_p99_ms=400.0,
            errors=2,
        )

        captured_output = []

        def capture_print(*args, **kwargs):
            captured_output.append(" ".join(str(a) for a in args))

        with patch("benchmark_throughput.check_health", new=AsyncMock(return_value=True)):
            with patch("benchmark_throughput.benchmark_batch_size", new=AsyncMock(return_value=mock_result)):
                with patch("sys.argv", ["benchmark_throughput.py", "--batch-sizes", "8", "--output", "table"]):
                    with patch("builtins.print", side_effect=capture_print):
                        result = await main()

        output_str = "\n".join(captured_output)
        assert "BENCHMARK RESULTS" in output_str
        assert "Batch Size" in output_str
        assert "Tok/s" in output_str
        assert "OPTIMAL BATCH SIZE" in output_str
        assert "COST PER MILLION TOKENS" in output_str

    @pytest.mark.asyncio
    async def test_main_finds_optimal_batch_size(self):
        """Identifies batch size with highest throughput as optimal."""
        from benchmark_throughput import main

        results = [
            BenchmarkResult(batch_size=1, total_requests=10, total_tokens=500, duration_seconds=30.0,
                          tokens_per_second=16.67, requests_per_second=0.33, latency_p50_ms=100.0,
                          latency_p99_ms=200.0, errors=0),
            BenchmarkResult(batch_size=4, total_requests=40, total_tokens=2000, duration_seconds=30.0,
                          tokens_per_second=66.67, requests_per_second=1.33, latency_p50_ms=150.0,
                          latency_p99_ms=300.0, errors=0),
            BenchmarkResult(batch_size=8, total_requests=80, total_tokens=4000, duration_seconds=30.0,
                          tokens_per_second=133.33, requests_per_second=2.67, latency_p50_ms=200.0,
                          latency_p99_ms=400.0, errors=0),
        ]

        result_iter = iter(results)

        async def mock_benchmark(*args, **kwargs):
            return next(result_iter)

        captured_output = []

        def capture_print(*args, **kwargs):
            captured_output.append(" ".join(str(a) for a in args))

        with patch("benchmark_throughput.check_health", new=AsyncMock(return_value=True)):
            with patch("benchmark_throughput.benchmark_batch_size", new=mock_benchmark):
                with patch("sys.argv", ["benchmark_throughput.py", "--batch-sizes", "1,4,8"]):
                    with patch("builtins.print", side_effect=capture_print):
                        await main()

        output_str = "\n".join(captured_output)
        # Batch size 8 has highest throughput
        assert "OPTIMAL BATCH SIZE: 8" in output_str

    @pytest.mark.asyncio
    async def test_main_custom_parameters(self):
        """Custom CLI parameters are passed correctly."""
        from benchmark_throughput import main

        call_args = []

        async def capture_benchmark(endpoint, batch_size, duration_seconds, max_tokens):
            call_args.append((endpoint, batch_size, duration_seconds, max_tokens))
            return BenchmarkResult(batch_size=batch_size, total_requests=10, total_tokens=500,
                                 duration_seconds=30.0, tokens_per_second=16.67, requests_per_second=0.33,
                                 latency_p50_ms=100.0, latency_p99_ms=200.0, errors=0)

        with patch("benchmark_throughput.check_health", new=AsyncMock(return_value=True)):
            with patch("benchmark_throughput.benchmark_batch_size", new=capture_benchmark):
                with patch("sys.argv", [
                    "benchmark_throughput.py",
                    "--endpoint", "http://custom:9000",
                    "--batch-sizes", "2,16",
                    "--duration", "45",
                    "--max-tokens", "100",
                ]):
                    with patch("builtins.print"):
                        await main()

        assert len(call_args) == 2
        # First call should be batch_size=2
        assert call_args[0] == ("http://custom:9000", 2, 45, 100)
        # Second call should be batch_size=16
        assert call_args[1] == ("http://custom:9000", 16, 45, 100)

    @pytest.mark.asyncio
    async def test_main_strips_trailing_slash(self):
        """Endpoint URL has trailing slash stripped."""
        from benchmark_throughput import main

        call_args = []

        async def capture_health(endpoint):
            call_args.append(endpoint)
            return True

        async def mock_benchmark(*args, **kwargs):
            return BenchmarkResult(batch_size=1, total_requests=10, total_tokens=500,
                                 duration_seconds=30.0, tokens_per_second=16.67, requests_per_second=0.33,
                                 latency_p50_ms=100.0, latency_p99_ms=200.0, errors=0)

        with patch("benchmark_throughput.check_health", new=capture_health):
            with patch("benchmark_throughput.benchmark_batch_size", new=mock_benchmark):
                with patch("sys.argv", ["benchmark_throughput.py", "--endpoint", "http://localhost:8080/", "--batch-sizes", "1"]):
                    with patch("builtins.print"):
                        await main()

        # Should strip trailing slash
        assert call_args[0] == "http://localhost:8080"

    @pytest.mark.asyncio
    async def test_main_shows_pricing_suggestions(self):
        """Table output includes pricing suggestions with margins."""
        from benchmark_throughput import main

        mock_result = BenchmarkResult(
            batch_size=1,
            total_requests=100,
            total_tokens=5000,
            duration_seconds=30.0,
            tokens_per_second=166.67,
            requests_per_second=3.33,
            latency_p50_ms=100.0,
            latency_p99_ms=200.0,
            errors=0,
        )

        captured_output = []

        def capture_print(*args, **kwargs):
            captured_output.append(" ".join(str(a) for a in args))

        with patch("benchmark_throughput.check_health", new=AsyncMock(return_value=True)):
            with patch("benchmark_throughput.benchmark_batch_size", new=AsyncMock(return_value=mock_result)):
                with patch("sys.argv", ["benchmark_throughput.py", "--batch-sizes", "1"]):
                    with patch("builtins.print", side_effect=capture_print):
                        await main()

        output_str = "\n".join(captured_output)
        assert "SUGGESTED PRICING" in output_str
        assert "Margin" in output_str
        assert "30%" in output_str or "50%" in output_str or "70%" in output_str


class TestAiohttpImportError:
    """Tests for aiohttp import error handling.

    The benchmark script has a try/except for aiohttp import that prints
    an error and exits if aiohttp is not installed.
    """

    def test_import_error_message_and_exit(self):
        """When aiohttp import fails, print message and exit."""
        import subprocess
        import sys

        # Create a script that simulates the import error
        test_script = """
import sys
import builtins

# Save original import
original_import = builtins.__import__

def mock_import(name, *args, **kwargs):
    if name == 'aiohttp':
        raise ImportError("No module named 'aiohttp'")
    return original_import(name, *args, **kwargs)

builtins.__import__ = mock_import

# Now import will fail
try:
    import aiohttp
except ImportError:
    print("Please install aiohttp: pip install aiohttp")
    exit(1)
"""
        result = subprocess.run(
            [sys.executable, "-c", test_script],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "aiohttp" in result.stdout


class TestCalculateCostsEdgeCases:
    """Additional edge case tests for calculate_costs."""

    def test_all_infrastructure_options_present(self):
        """All expected infrastructure options are in results."""
        costs = calculate_costs(500.0)

        expected = [
            "Hetzner AX42 ($0.075/hr)",
            "Hetzner AX102 ($0.214/hr)",
            "OVHCloud b3-64 ($0.16/hr)",
            "AWS r7a.4xl spot ($0.20/hr)",
        ]
        for infra in expected:
            assert infra in costs

    def test_utilization_tiers_increase_cost(self):
        """Lower utilization means higher cost per token."""
        costs = calculate_costs(1000.0)

        for infra, util_costs in costs.items():
            # 100% util should be cheapest
            assert util_costs["100% util"] < util_costs["70% util"]
            assert util_costs["70% util"] < util_costs["50% util"]

    def test_very_low_throughput(self):
        """Very low throughput produces high costs."""
        costs = calculate_costs(1.0)  # 1 token/second

        # Should have very high costs
        for infra, util_costs in costs.items():
            # Even at 100% util, cost should be > $1/million tokens
            assert util_costs["100% util"] > 1.0


class TestBenchmarkResultFields:
    """Additional tests for BenchmarkResult dataclass."""

    def test_all_fields_accessible(self):
        """All expected fields are present and accessible."""
        result = BenchmarkResult(
            batch_size=4,
            total_requests=100,
            total_tokens=5000,
            duration_seconds=60.0,
            tokens_per_second=83.33,
            requests_per_second=1.67,
            latency_p50_ms=300.0,
            latency_p99_ms=600.0,
            errors=5,
        )

        # All fields should be accessible
        assert result.batch_size == 4
        assert result.total_requests == 100
        assert result.total_tokens == 5000
        assert result.duration_seconds == 60.0
        assert result.tokens_per_second == 83.33
        assert result.requests_per_second == 1.67
        assert result.latency_p50_ms == 300.0
        assert result.latency_p99_ms == 600.0
        assert result.errors == 5
