"""Tests for the cost tracker module."""

import pytest
from pathlib import Path
import sys

# Add benchmark module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmark"))

from benchmark.cost_tracker import CostTracker, CostMetrics


class TestCostMetrics:
    """Tests for CostMetrics dataclass."""

    def test_tokens_per_hour(self):
        """Test tokens per hour calculation."""
        metrics = CostMetrics(
            tokens_per_second=100,
            hardware_cost_per_hour=0.39,
        )
        assert metrics.tokens_per_hour == 360000  # 100 * 3600

    def test_cost_per_million_tokens(self):
        """Test cost per million tokens calculation."""
        metrics = CostMetrics(
            tokens_per_second=100,
            hardware_cost_per_hour=0.39,
        )
        # $0.39/hr / 360000 tokens/hr * 1M = $1.0833
        expected = (0.39 / 360000) * 1_000_000
        assert abs(metrics.cost_per_million_tokens - expected) < 0.0001

    def test_cost_per_million_zero_throughput(self):
        """Test cost calculation with zero throughput."""
        metrics = CostMetrics(
            tokens_per_second=0,
            hardware_cost_per_hour=0.39,
        )
        assert metrics.cost_per_million_tokens == float("inf")

    def test_at_utilization(self):
        """Test utilization-adjusted cost calculation."""
        metrics = CostMetrics(
            tokens_per_second=100,
            hardware_cost_per_hour=0.39,
        )
        cost_100 = metrics.cost_per_million_tokens
        cost_70 = metrics.at_utilization(0.7)
        cost_50 = metrics.at_utilization(0.5)

        # Lower utilization = higher effective cost
        assert cost_70 > cost_100
        assert cost_50 > cost_70
        assert abs(cost_70 - cost_100 / 0.7) < 0.0001
        assert abs(cost_50 - cost_100 / 0.5) < 0.0001

    def test_at_utilization_zero(self):
        """Test utilization at 0% (edge case)."""
        metrics = CostMetrics(
            tokens_per_second=100,
            hardware_cost_per_hour=0.39,
        )
        assert metrics.at_utilization(0) == float("inf")

    def test_to_dict(self):
        """Test dictionary serialization."""
        metrics = CostMetrics(
            tokens_per_second=100,
            hardware_cost_per_hour=0.39,
            hardware_name="NVIDIA A40",
        )
        d = metrics.to_dict()

        assert d["hardware_name"] == "NVIDIA A40"
        assert d["hardware_cost_per_hour"] == 0.39
        assert d["tokens_per_second"] == 100
        assert "cost_per_million_tokens" in d


class TestCostTracker:
    """Tests for CostTracker class."""

    @pytest.fixture
    def tracker(self):
        """Create a cost tracker with default config."""
        return CostTracker()

    def test_load_config(self, tracker):
        """Test that config loads successfully."""
        assert tracker.pricing is not None
        assert "runpod" in tracker.pricing

    def test_get_hardware_price_gpu(self, tracker):
        """Test getting GPU price."""
        price, name = tracker.get_hardware_price("a40", "runpod", use_spot=True)
        assert price == 0.39
        assert "A40" in name

    def test_get_hardware_price_on_demand(self, tracker):
        """Test getting on-demand price."""
        spot_price, _ = tracker.get_hardware_price("a40", "runpod", use_spot=True)
        on_demand_price, _ = tracker.get_hardware_price("a40", "runpod", use_spot=False)

        assert on_demand_price > spot_price
        assert on_demand_price == 0.69

    def test_get_hardware_price_cpu(self, tracker):
        """Test getting CPU price."""
        price, name = tracker.get_hardware_price("cpu_64", "runpod")
        assert price == 0.24
        assert "64" in name

    def test_get_hardware_price_unknown(self, tracker):
        """Test error for unknown hardware."""
        with pytest.raises(ValueError, match="Unknown hardware"):
            tracker.get_hardware_price("unknown_gpu", "runpod")

    def test_calculate_cost(self, tracker):
        """Test cost calculation."""
        metrics = tracker.calculate_cost(
            hardware_id="a40",
            tokens_per_second=100,
            provider="runpod",
            use_spot=True,
        )

        assert metrics.hardware_cost_per_hour == 0.39
        assert metrics.tokens_per_second == 100
        assert metrics.cost_per_million_tokens > 0

    def test_compare_hardware(self, tracker):
        """Test hardware comparison."""
        results = tracker.compare_hardware(
            tokens_per_second=100,
            hardware_ids=["a40", "cpu_64"],
            provider="runpod",
        )

        assert "a40" in results
        assert "cpu_64" in results

        # CPU should be cheaper per hour but may be slower
        assert results["cpu_64"].hardware_cost_per_hour < results["a40"].hardware_cost_per_hour

    def test_format_cost_table(self, tracker):
        """Test markdown table generation."""
        results = tracker.compare_hardware(
            tokens_per_second=100,
            hardware_ids=["a40", "cpu_64"],
        )

        table = tracker.format_cost_table(results)

        assert "| Hardware |" in table
        assert "A40" in table or "a40" in table
        assert "$" in table

    def test_get_utilization_levels(self, tracker):
        """Test utilization level retrieval."""
        levels = tracker.get_utilization_levels()

        assert 1.0 in levels
        assert 0.7 in levels
        assert 0.5 in levels


class TestCostComparisons:
    """Integration tests comparing costs across configurations."""

    @pytest.fixture
    def tracker(self):
        return CostTracker()

    def test_gpu_vs_cpu_cost_efficiency(self, tracker):
        """Test that CPU can be more cost-efficient for some workloads."""
        # GPU: faster but more expensive
        gpu_metrics = tracker.calculate_cost("a40", tokens_per_second=200)

        # CPU: slower but cheaper
        cpu_metrics = tracker.calculate_cost("cpu_64", tokens_per_second=50)

        # Both should have valid costs
        assert gpu_metrics.cost_per_million_tokens > 0
        assert cpu_metrics.cost_per_million_tokens > 0

        # At these rates, compare cost efficiency
        # GPU: $0.39/hr @ 200 tok/s = $0.39 / (200 * 3600) * 1M = $0.542/1M
        # CPU: $0.24/hr @ 50 tok/s = $0.24 / (50 * 3600) * 1M = $1.333/1M
        # GPU is more cost efficient here
        assert gpu_metrics.cost_per_million_tokens < cpu_metrics.cost_per_million_tokens

    def test_spot_vs_on_demand_savings(self, tracker):
        """Test spot pricing savings."""
        spot_price, _ = tracker.get_hardware_price("a40", use_spot=True)
        on_demand_price, _ = tracker.get_hardware_price("a40", use_spot=False)

        savings_pct = (on_demand_price - spot_price) / on_demand_price * 100

        # Expect significant savings with spot
        assert savings_pct > 30  # At least 30% savings
