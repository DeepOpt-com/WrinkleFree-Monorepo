"""
Cost tracking for inference benchmarking.

Calculates cost per million tokens based on hardware pricing and throughput.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class CostMetrics:
    """Cost metrics for a specific throughput measurement."""

    tokens_per_second: float
    hardware_cost_per_hour: float
    hardware_name: str = ""

    @property
    def tokens_per_hour(self) -> float:
        """Tokens generated per hour at current throughput."""
        return self.tokens_per_second * 3600

    @property
    def cost_per_million_tokens(self) -> float:
        """Cost per 1M tokens at 100% utilization."""
        if self.tokens_per_hour == 0:
            return float("inf")
        return (self.hardware_cost_per_hour / self.tokens_per_hour) * 1_000_000

    def at_utilization(self, utilization: float) -> float:
        """Cost per 1M tokens at given utilization level."""
        if utilization <= 0:
            return float("inf")
        return self.cost_per_million_tokens / utilization

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "hardware_name": self.hardware_name,
            "hardware_cost_per_hour": self.hardware_cost_per_hour,
            "tokens_per_second": round(self.tokens_per_second, 2),
            "tokens_per_hour": round(self.tokens_per_hour, 0),
            "cost_per_million_tokens": {
                "100%_util": round(self.cost_per_million_tokens, 6),
                "70%_util": round(self.at_utilization(0.7), 6),
                "50%_util": round(self.at_utilization(0.5), 6),
            },
        }


class CostTracker:
    """Tracks and calculates costs for different hardware configurations."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize cost tracker with hardware pricing config.

        Args:
            config_path: Path to hardware.yaml config file.
                        If None, uses default config location.
        """
        if config_path is None:
            config_path = Path(__file__).parent / "configs" / "hardware.yaml"

        self.config_path = Path(config_path)
        self.pricing = self._load_config()

    def _load_config(self) -> dict:
        """Load hardware pricing configuration."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Hardware config not found: {self.config_path}")

        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def get_hardware_price(
        self,
        hardware_id: str,
        provider: str = "runpod",
        use_spot: bool = True,
    ) -> tuple[float, str]:
        """
        Get hourly price for a hardware configuration.

        Args:
            hardware_id: Hardware identifier (e.g., "a40", "cpu_32")
            provider: Cloud provider (default: "runpod")
            use_spot: Use spot/preemptible pricing if available

        Returns:
            Tuple of (price_per_hour, hardware_name)
        """
        provider_config = self.pricing.get(provider, {})

        # Check GPU configs
        if hardware_id in provider_config.get("gpu", {}):
            hw = provider_config["gpu"][hardware_id]
            price_key = "spot_price_per_hour" if use_spot else "on_demand_price_per_hour"
            return hw.get(price_key, hw.get("price_per_hour", 0)), hw.get("name", hardware_id)

        # Check CPU configs
        if hardware_id in provider_config.get("cpu", {}):
            hw = provider_config["cpu"][hardware_id]
            return hw.get("price_per_hour", 0), hw.get("name", hardware_id)

        # Check top-level configs (for providers like hetzner)
        if hardware_id in provider_config:
            hw = provider_config[hardware_id]
            return hw.get("price_per_hour", 0), hw.get("name", hardware_id)

        raise ValueError(f"Unknown hardware: {hardware_id} for provider {provider}")

    def calculate_cost(
        self,
        hardware_id: str,
        tokens_per_second: float,
        provider: str = "runpod",
        use_spot: bool = True,
    ) -> CostMetrics:
        """
        Calculate cost metrics for a benchmark result.

        Args:
            hardware_id: Hardware identifier
            tokens_per_second: Measured throughput
            provider: Cloud provider
            use_spot: Use spot pricing

        Returns:
            CostMetrics with calculated values
        """
        price, name = self.get_hardware_price(hardware_id, provider, use_spot)

        return CostMetrics(
            tokens_per_second=tokens_per_second,
            hardware_cost_per_hour=price,
            hardware_name=name,
        )

    def compare_hardware(
        self,
        tokens_per_second: float,
        hardware_ids: Optional[list[str]] = None,
        provider: str = "runpod",
    ) -> dict[str, CostMetrics]:
        """
        Compare costs across multiple hardware configurations.

        Args:
            tokens_per_second: Assumed throughput (same for all)
            hardware_ids: List of hardware IDs to compare
            provider: Cloud provider

        Returns:
            Dictionary mapping hardware_id to CostMetrics
        """
        if hardware_ids is None:
            # Default to all available hardware
            provider_config = self.pricing.get(provider, {})
            hardware_ids = []
            hardware_ids.extend(provider_config.get("gpu", {}).keys())
            hardware_ids.extend(provider_config.get("cpu", {}).keys())

        results = {}
        for hw_id in hardware_ids:
            try:
                results[hw_id] = self.calculate_cost(
                    hw_id, tokens_per_second, provider, use_spot=True
                )
            except ValueError:
                continue

        return results

    def format_cost_table(self, results: dict[str, CostMetrics]) -> str:
        """
        Format cost comparison as a markdown table.

        Args:
            results: Dictionary of hardware_id -> CostMetrics

        Returns:
            Markdown table string
        """
        lines = [
            "| Hardware | $/hr | Tok/s | $/1M @ 100% | $/1M @ 70% | $/1M @ 50% |",
            "|----------|------|-------|-------------|------------|------------|",
        ]

        # Sort by cost per million tokens
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].cost_per_million_tokens,
        )

        for hw_id, metrics in sorted_results:
            lines.append(
                f"| {metrics.hardware_name} | "
                f"${metrics.hardware_cost_per_hour:.2f} | "
                f"{metrics.tokens_per_second:.1f} | "
                f"${metrics.cost_per_million_tokens:.4f} | "
                f"${metrics.at_utilization(0.7):.4f} | "
                f"${metrics.at_utilization(0.5):.4f} |"
            )

        return "\n".join(lines)

    def get_utilization_levels(self) -> list[float]:
        """Get configured utilization levels for cost calculations."""
        return self.pricing.get("utilization_levels", [1.0, 0.7, 0.5])
