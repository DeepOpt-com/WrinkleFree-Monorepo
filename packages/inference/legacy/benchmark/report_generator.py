"""
Report generator for cost benchmarks.

Generates markdown reports with comparison tables and charts.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
import json
import logging

from benchmark.metrics import CostBenchmarkResult

logger = logging.getLogger(__name__)


@dataclass
class ReportConfig:
    """Configuration for report generation."""

    title: str = "BitNet Cost Benchmark Report"
    include_charts: bool = True
    include_raw_data: bool = False
    utilization_levels: list[float] = None

    def __post_init__(self):
        if self.utilization_levels is None:
            self.utilization_levels = [1.0, 0.7, 0.5]


class ReportGenerator:
    """
    Generates markdown reports from benchmark results.

    Usage:
        results = [result1, result2, ...]
        generator = ReportGenerator(results)
        generator.generate_report(output_path)
    """

    def __init__(
        self,
        results: list[CostBenchmarkResult],
        config: Optional[ReportConfig] = None,
    ):
        self.results = results
        self.config = config or ReportConfig()

    def generate_report(self, output_path: Path) -> Path:
        """
        Generate full markdown report.

        Args:
            output_path: Path to save the report

        Returns:
            Path to generated report
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sections = [
            self._generate_header(),
            self._generate_summary(),
            self._generate_comparison_table(),
            self._generate_cost_analysis(),
            self._generate_hardware_comparison(),
            self._generate_methodology(),
        ]

        if self.config.include_charts:
            sections.append(self._generate_chart_section())

        if self.config.include_raw_data:
            sections.append(self._generate_raw_data())

        report = "\n\n".join(sections)

        with open(output_path, "w") as f:
            f.write(report)

        logger.info(f"Report saved to: {output_path}")
        return output_path

    def _generate_header(self) -> str:
        """Generate report header."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"""# {self.config.title}

Generated: {timestamp}

---"""

    def _generate_summary(self) -> str:
        """Generate executive summary."""
        if not self.results:
            return "## Summary\n\nNo benchmark results available."

        # Find best result by cost efficiency
        best_cost = min(self.results, key=lambda r: r.cost_per_million_tokens or float("inf"))
        best_throughput = max(self.results, key=lambda r: r.tokens_per_second)

        models = set(r.model for r in self.results)
        hardware = set(r.hardware for r in self.results)

        return f"""## Summary

**Benchmarks Run:** {len(self.results)}
**Models Tested:** {', '.join(models)}
**Hardware Tested:** {', '.join(hardware)}

### Key Findings

| Metric | Best Result | Configuration |
|--------|-------------|---------------|
| Lowest Cost | ${best_cost.cost_per_million_tokens:.4f}/1M tokens | {best_cost.model} on {best_cost.hardware} |
| Highest Throughput | {best_throughput.tokens_per_second:.1f} tok/s | {best_throughput.model} on {best_throughput.hardware} |"""

    def _generate_comparison_table(self) -> str:
        """Generate main comparison table."""
        if not self.results:
            return ""

        lines = [
            "## Performance Comparison",
            "",
            "| Model | Quant | Hardware | Tok/s | TTFT P50 | Lat P99 | Memory | $/1M Tokens |",
            "|-------|-------|----------|-------|----------|---------|--------|-------------|",
        ]

        # Sort by cost per million tokens
        sorted_results = sorted(
            self.results,
            key=lambda r: r.cost_per_million_tokens or float("inf"),
        )

        for r in sorted_results:
            lines.append(
                f"| {r.model} | {r.quantization} | {r.hardware} | "
                f"{r.tokens_per_second:.1f} | {r.ttft_p50_ms:.0f}ms | {r.latency_p99_ms:.0f}ms | "
                f"{r.memory_usage_gb:.1f}GB | ${r.cost_per_million_tokens:.4f} |"
            )

        return "\n".join(lines)

    def _generate_cost_analysis(self) -> str:
        """Generate cost analysis section."""
        if not self.results:
            return ""

        lines = [
            "## Cost Analysis",
            "",
            "Cost per million tokens at different utilization levels:",
            "",
            "| Model | Hardware | 100% Util | 70% Util | 50% Util |",
            "|-------|----------|-----------|----------|----------|",
        ]

        for r in self.results:
            cost_100 = r.cost_per_million_tokens
            cost_70 = r.cost_per_million_at_70pct
            cost_50 = r.cost_per_million_at_50pct

            lines.append(
                f"| {r.model} | {r.hardware} | "
                f"${cost_100:.4f} | ${cost_70:.4f} | ${cost_50:.4f} |"
            )

        # Add comparison to commercial APIs
        lines.extend([
            "",
            "### Comparison to Commercial APIs",
            "",
            "| Provider | Model | $/1M Tokens (Input) | $/1M Tokens (Output) |",
            "|----------|-------|---------------------|----------------------|",
            "| OpenAI | GPT-4 Turbo | $10.00 | $30.00 |",
            "| OpenAI | GPT-3.5 Turbo | $0.50 | $1.50 |",
            "| Anthropic | Claude 3 Opus | $15.00 | $75.00 |",
            "| Anthropic | Claude 3 Sonnet | $3.00 | $15.00 |",
            "",
            "*Note: Self-hosted costs do not include infrastructure setup, maintenance, or engineering time.*",
        ])

        return "\n".join(lines)

    def _generate_hardware_comparison(self) -> str:
        """Generate hardware comparison section."""
        if not self.results:
            return ""

        # Group by hardware
        by_hardware: dict[str, list[CostBenchmarkResult]] = {}
        for r in self.results:
            if r.hardware not in by_hardware:
                by_hardware[r.hardware] = []
            by_hardware[r.hardware].append(r)

        lines = [
            "## Hardware Comparison",
            "",
        ]

        for hw, results in by_hardware.items():
            avg_throughput = sum(r.tokens_per_second for r in results) / len(results)
            avg_cost = sum(r.cost_per_million_tokens for r in results) / len(results)
            hw_cost = results[0].hardware_cost_per_hour

            lines.extend([
                f"### {hw}",
                f"- **Hourly Cost:** ${hw_cost:.2f}",
                f"- **Avg Throughput:** {avg_throughput:.1f} tok/s",
                f"- **Avg Cost/1M Tokens:** ${avg_cost:.4f}",
                "",
            ])

        return "\n".join(lines)

    def _generate_methodology(self) -> str:
        """Generate methodology section."""
        return """## Methodology

### Benchmark Configuration

- **Warmup Requests:** 5-10 requests before timing
- **Benchmark Duration:** 60 seconds per batch size
- **Batch Sizes Tested:** 1, 4, 8, 16
- **Max Tokens:** 100 tokens per request

### Metrics Collected

- **Throughput:** Tokens generated per second (tok/s)
- **TTFT:** Time to first token (P50, P99)
- **Latency:** End-to-end request latency (P50, P95, P99)
- **Memory:** Peak memory usage during benchmark

### Cost Calculation

```
Cost per 1M tokens = (Hourly Hardware Cost / Tokens per Hour) * 1,000,000
```

Utilization-adjusted costs account for realistic utilization:
- 100% utilization: Theoretical maximum
- 70% utilization: Realistic production workload
- 50% utilization: Conservative estimate

### Quantization Methods

- **Native BitNet:** Models trained from scratch with 1.58-bit weights
- **Naive Conversion:** FP16 weights rounded to ternary (-1, 0, 1)
  - *Warning: Naive conversion produces poor quality outputs*
  - *Only suitable for speed/cost benchmarking, not production use*"""

    def _generate_chart_section(self) -> str:
        """Generate chart placeholder section."""
        return """## Charts

*Charts generated separately using matplotlib. See `results/charts/` directory.*

Recommended charts:
1. Throughput vs Hardware (bar chart)
2. Cost vs Throughput (scatter plot)
3. Latency Distribution (box plot)
4. Cost at Different Utilization Levels (grouped bar chart)"""

    def _generate_raw_data(self) -> str:
        """Generate raw data section."""
        lines = [
            "## Raw Data",
            "",
            "```json",
            json.dumps([r.to_dict() for r in self.results], indent=2),
            "```",
        ]
        return "\n".join(lines)


def generate_summary_from_dir(results_dir: Path, output_path: Path) -> Path:
    """
    Generate summary report from all results in a directory.

    Args:
        results_dir: Directory containing JSON result files
        output_path: Path to save the summary report

    Returns:
        Path to generated report
    """
    results_dir = Path(results_dir)
    results = []

    for json_file in results_dir.glob("*.json"):
        try:
            result = CostBenchmarkResult.load(json_file)
            results.append(result)
        except Exception as e:
            logger.warning(f"Failed to load {json_file}: {e}")

    if not results:
        logger.warning(f"No valid results found in {results_dir}")

    generator = ReportGenerator(results)
    return generator.generate_report(output_path)
