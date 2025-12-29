"""
Metrics dataclasses for cost benchmarking.

Extends the existing benchmark metrics with cost tracking, TTFT, and memory usage.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import json
from pathlib import Path


@dataclass
class BenchmarkMetrics:
    """Core metrics from a benchmark run."""

    name: str
    requests: int
    successful: int
    failed: int
    total_time_seconds: float
    tokens_generated: int

    # Latency percentiles (ms)
    latency_avg_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_min_ms: float
    latency_max_ms: float

    # Time to first token (ms)
    ttft_avg_ms: float = 0.0
    ttft_p50_ms: float = 0.0
    ttft_p95_ms: float = 0.0
    ttft_p99_ms: float = 0.0

    # Throughput
    requests_per_second: float = 0.0
    tokens_per_second: float = 0.0

    # Resource usage
    memory_usage_gb: float = 0.0
    peak_memory_gb: float = 0.0
    cpu_utilization_percent: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None

    # Memory bandwidth (from memory_profiler.py)
    memory_bandwidth_gb_per_sec: float = 0.0
    memory_bandwidth_utilization_percent: float = 0.0

    @classmethod
    def from_latencies(
        cls,
        name: str,
        latencies: list[float],
        tokens: list[int],
        total_time: float,
        failed: int = 0,
        ttfts: Optional[list[float]] = None,
        memory_usage_gb: float = 0.0,
    ) -> "BenchmarkMetrics":
        """Create metrics from raw latency data."""
        import statistics

        if not latencies:
            return cls(
                name=name,
                requests=failed,
                successful=0,
                failed=failed,
                total_time_seconds=total_time,
                tokens_generated=0,
                latency_avg_ms=0,
                latency_p50_ms=0,
                latency_p95_ms=0,
                latency_p99_ms=0,
                latency_min_ms=0,
                latency_max_ms=0,
                requests_per_second=0,
                tokens_per_second=0,
                memory_usage_gb=memory_usage_gb,
            )

        sorted_lat = sorted(latencies)
        total_tokens = sum(tokens)

        # Calculate TTFT metrics if available
        ttft_avg = ttft_p50 = ttft_p95 = ttft_p99 = 0.0
        if ttfts and len(ttfts) > 0:
            sorted_ttft = sorted(ttfts)
            ttft_avg = statistics.mean(ttfts)
            ttft_p50 = statistics.median(ttfts)
            ttft_p95 = sorted_ttft[int(len(sorted_ttft) * 0.95)] if len(sorted_ttft) > 1 else sorted_ttft[0]
            ttft_p99 = sorted_ttft[int(len(sorted_ttft) * 0.99)] if len(sorted_ttft) > 1 else sorted_ttft[0]

        return cls(
            name=name,
            requests=len(latencies) + failed,
            successful=len(latencies),
            failed=failed,
            total_time_seconds=total_time,
            tokens_generated=total_tokens,
            latency_avg_ms=statistics.mean(latencies),
            latency_p50_ms=statistics.median(latencies),
            latency_p95_ms=sorted_lat[int(len(sorted_lat) * 0.95)] if len(sorted_lat) > 1 else sorted_lat[0],
            latency_p99_ms=sorted_lat[int(len(sorted_lat) * 0.99)] if len(sorted_lat) > 1 else sorted_lat[0],
            latency_min_ms=min(latencies),
            latency_max_ms=max(latencies),
            ttft_avg_ms=ttft_avg,
            ttft_p50_ms=ttft_p50,
            ttft_p95_ms=ttft_p95,
            ttft_p99_ms=ttft_p99,
            requests_per_second=len(latencies) / total_time if total_time > 0 else 0,
            tokens_per_second=total_tokens / total_time if total_time > 0 else 0,
            memory_usage_gb=memory_usage_gb,
        )


@dataclass
class CostBenchmarkResult:
    """Complete result from a cost benchmark run."""

    # Identification
    run_id: str
    timestamp: datetime = field(default_factory=datetime.now)

    # Configuration
    model: str = ""
    model_size_params: str = ""
    quantization: str = ""  # "native" or "naive"
    bits_per_weight: float = 1.58
    hardware: str = ""
    hardware_type: str = ""  # "gpu" or "cpu"

    # Performance metrics
    metrics: Optional[BenchmarkMetrics] = None
    tokens_per_second: float = 0.0
    ttft_p50_ms: float = 0.0
    ttft_p99_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0

    # Resource usage
    memory_usage_gb: float = 0.0
    peak_memory_gb: float = 0.0
    model_size_gb: float = 0.0

    # Cost metrics
    hardware_cost_per_hour: float = 0.0
    cost_per_million_tokens: float = 0.0
    cost_per_million_at_70pct: float = 0.0  # At 70% utilization
    cost_per_million_at_50pct: float = 0.0  # At 50% utilization

    # Benchmark details
    total_requests: int = 0
    total_tokens_generated: int = 0
    duration_seconds: float = 0.0
    errors: int = 0
    warmup_requests: int = 0
    batch_sizes_tested: list[int] = field(default_factory=list)
    optimal_batch_size: int = 0

    def calculate_costs(self) -> None:
        """Calculate cost metrics from throughput and hardware cost."""
        if self.tokens_per_second > 0:
            tokens_per_hour = self.tokens_per_second * 3600
            self.cost_per_million_tokens = (self.hardware_cost_per_hour / tokens_per_hour) * 1_000_000
            self.cost_per_million_at_70pct = self.cost_per_million_tokens / 0.7
            self.cost_per_million_at_50pct = self.cost_per_million_tokens / 0.5

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
            "model": self.model,
            "model_size_params": self.model_size_params,
            "quantization": self.quantization,
            "bits_per_weight": self.bits_per_weight,
            "hardware": self.hardware,
            "hardware_type": self.hardware_type,
            "tokens_per_second": round(self.tokens_per_second, 2),
            "ttft_p50_ms": round(self.ttft_p50_ms, 1),
            "ttft_p99_ms": round(self.ttft_p99_ms, 1),
            "latency_p50_ms": round(self.latency_p50_ms, 1),
            "latency_p95_ms": round(self.latency_p95_ms, 1),
            "latency_p99_ms": round(self.latency_p99_ms, 1),
            "memory_usage_gb": round(self.memory_usage_gb, 2),
            "peak_memory_gb": round(self.peak_memory_gb, 2),
            "model_size_gb": round(self.model_size_gb, 2),
            "hardware_cost_per_hour": self.hardware_cost_per_hour,
            "cost_per_million_tokens": round(self.cost_per_million_tokens, 6),
            "cost_per_million_at_70pct": round(self.cost_per_million_at_70pct, 6),
            "cost_per_million_at_50pct": round(self.cost_per_million_at_50pct, 6),
            "total_requests": self.total_requests,
            "total_tokens_generated": self.total_tokens_generated,
            "duration_seconds": round(self.duration_seconds, 2),
            "errors": self.errors,
            "warmup_requests": self.warmup_requests,
            "batch_sizes_tested": self.batch_sizes_tested,
            "optimal_batch_size": self.optimal_batch_size,
        }

    def save(self, output_dir: Path) -> Path:
        """Save result to JSON file."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{self.timestamp.strftime('%Y%m%d_%H%M%S')}_{self.model}_{self.quantization}_{self.hardware}.json"
        filepath = output_dir / filename

        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        return filepath

    @classmethod
    def load(cls, filepath: Path) -> "CostBenchmarkResult":
        """Load result from JSON file."""
        with open(filepath) as f:
            data = json.load(f)

        result = cls(
            run_id=data["run_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )

        # Copy all other fields
        for key, value in data.items():
            if key not in ("run_id", "timestamp") and hasattr(result, key):
                setattr(result, key, value)

        return result
