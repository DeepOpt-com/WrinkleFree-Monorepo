#!/usr/bin/env python3
"""
Cost benchmarking script for BitNet inference.

Runs benchmarks against an inference server and calculates cost metrics.

Usage:
    # Against local server
    python scripts/benchmark_cost.py --url http://localhost:8080 --hardware a40

    # With specific model
    python scripts/benchmark_cost.py --url http://localhost:8080 --hardware cpu_64 --model bitnet-2b-4t

    # Full benchmark suite
    python scripts/benchmark_cost.py --url http://localhost:8080 --hardware a40 --full
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from benchmark.runner import BenchmarkConfig, BenchmarkRunner
from benchmark.cost_tracker import CostTracker
from benchmark.report_generator import ReportGenerator, generate_summary_from_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run cost benchmarks for BitNet inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--url",
        default="http://localhost:8080",
        help="Inference server URL (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--hardware",
        required=True,
        choices=["a40", "l4", "cpu_16", "cpu_32", "cpu_64"],
        help="Hardware configuration being benchmarked",
    )
    parser.add_argument(
        "--model",
        default="bitnet-2b-4t",
        help="Model being benchmarked (default: bitnet-2b-4t)",
    )
    parser.add_argument(
        "--model-size",
        default="2B",
        help="Model size for reporting (default: 2B)",
    )
    parser.add_argument(
        "--quantization",
        default="native",
        choices=["native", "naive"],
        help="Quantization type (default: native)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/raw"),
        help="Output directory for results (default: results/raw)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,4,8,16",
        help="Comma-separated batch sizes to test (default: 1,4,8,16)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Duration per batch size in seconds (default: 60)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup requests (default: 5)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Max tokens per request (default: 100)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full benchmark suite with all batch sizes",
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate summary report after benchmarking",
    )
    parser.add_argument(
        "--output",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )

    return parser.parse_args()


async def run_benchmark(args) -> dict:
    """Run the benchmark with given arguments."""
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]

    if args.full:
        batch_sizes = [1, 2, 4, 8, 16, 32]

    # Determine hardware type
    hardware_type = "gpu" if args.hardware in ["a40", "l4", "a100_40gb"] else "cpu"

    config = BenchmarkConfig(
        model=args.model,
        model_size=args.model_size,
        quantization=args.quantization,
        hardware=args.hardware,
        hardware_type=hardware_type,
        batch_sizes=batch_sizes,
        warmup_requests=args.warmup,
        benchmark_requests=50,
        max_tokens=args.max_tokens,
        duration_seconds=args.duration,
        server_url=args.url,
    )

    logger.info(f"Starting benchmark: {args.model} on {args.hardware}")
    logger.info(f"Server URL: {args.url}")
    logger.info(f"Batch sizes: {batch_sizes}")

    runner = BenchmarkRunner(config)
    result = await runner.run()

    # Save result
    output_path = result.save(args.output_dir)
    logger.info(f"Result saved to: {output_path}")

    return result.to_dict()


def print_table(result: dict):
    """Print result as formatted table."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    print(f"\nModel: {result['model']}")
    print(f"Hardware: {result['hardware']} ({result['hardware_type']})")
    print(f"Quantization: {result['quantization']}")

    print("\n--- Performance ---")
    print(f"Throughput: {result['tokens_per_second']:.1f} tokens/sec")
    print(f"TTFT P50: {result['ttft_p50_ms']:.0f}ms")
    print(f"TTFT P99: {result['ttft_p99_ms']:.0f}ms")
    print(f"Latency P50: {result['latency_p50_ms']:.0f}ms")
    print(f"Latency P99: {result['latency_p99_ms']:.0f}ms")

    print("\n--- Cost Analysis ---")
    print(f"Hardware Cost: ${result['hardware_cost_per_hour']:.2f}/hr")
    print(f"Cost per 1M tokens (100% util): ${result['cost_per_million_tokens']:.4f}")
    print(f"Cost per 1M tokens (70% util): ${result['cost_per_million_at_70pct']:.4f}")
    print(f"Cost per 1M tokens (50% util): ${result['cost_per_million_at_50pct']:.4f}")

    print("\n--- Resource Usage ---")
    print(f"Memory Usage: {result['memory_usage_gb']:.1f} GB")
    print(f"Optimal Batch Size: {result['optimal_batch_size']}")

    print("\n" + "=" * 80)


def print_json(result: dict):
    """Print result as JSON."""
    print(json.dumps(result, indent=2))


async def main():
    args = parse_args()

    try:
        result = await run_benchmark(args)

        if args.output == "table":
            print_table(result)
        else:
            print_json(result)

        # Generate summary report if requested
        if args.generate_report:
            report_path = args.output_dir.parent / "reports" / "summary.md"
            generate_summary_from_dir(args.output_dir, report_path)
            logger.info(f"Summary report: {report_path}")

        return 0

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
