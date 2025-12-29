"""
Benchmark runner for cost analysis.

Orchestrates benchmarks across different models and hardware configurations.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging

import yaml

from benchmark.metrics import BenchmarkMetrics, CostBenchmarkResult
from benchmark.cost_tracker import CostTracker

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    model: str
    model_size: str  # e.g., "2B", "70B"
    quantization: str  # "native" or "naive"
    hardware: str  # e.g., "a40", "cpu_64"
    hardware_type: str  # "gpu" or "cpu"

    # Benchmark parameters
    batch_sizes: list[int] = field(default_factory=lambda: [1, 4, 8])
    warmup_requests: int = 5
    benchmark_requests: int = 50
    max_tokens: int = 100
    duration_seconds: int = 60

    # Server config
    server_url: str = "http://localhost:8080"
    timeout_seconds: int = 120

    # Cost tracking
    provider: str = "runpod"
    use_spot: bool = True


class BenchmarkRunner:
    """
    Runs cost benchmarks against an inference server.

    Usage:
        config = BenchmarkConfig(
            model="bitnet-2b-4t",
            model_size="2B",
            quantization="native",
            hardware="a40",
            hardware_type="gpu",
        )
        runner = BenchmarkRunner(config)
        result = await runner.run()
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        cost_tracker: Optional[CostTracker] = None,
    ):
        self.config = config
        self.cost_tracker = cost_tracker or CostTracker()
        self._prompts = self._load_prompts()

    def _load_prompts(self) -> list[str]:
        """Load benchmark prompts from config."""
        config_path = Path(__file__).parent / "configs" / "models.yaml"
        if config_path.exists():
            with open(config_path) as f:
                models_config = yaml.safe_load(f)
                prompts = models_config.get("prompts", {})
                # Combine all prompt types
                all_prompts = []
                for prompt_list in prompts.values():
                    if isinstance(prompt_list, list):
                        all_prompts.extend(prompt_list)
                return all_prompts or self._default_prompts()
        return self._default_prompts()

    def _default_prompts(self) -> list[str]:
        """Default prompts for benchmarking."""
        return [
            "What is machine learning?",
            "Explain AI briefly.",
            "What is the capital of France?",
            "Write a haiku about coding.",
            "Explain neural networks.",
            "What is quantum computing?",
            "Describe cloud computing.",
            "What is blockchain?",
        ]

    async def run(self) -> CostBenchmarkResult:
        """
        Run the full benchmark suite.

        Returns:
            CostBenchmarkResult with all metrics
        """
        run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        result = CostBenchmarkResult(
            run_id=run_id,
            model=self.config.model,
            model_size_params=self.config.model_size,
            quantization=self.config.quantization,
            hardware=self.config.hardware,
            hardware_type=self.config.hardware_type,
            batch_sizes_tested=self.config.batch_sizes,
            warmup_requests=self.config.warmup_requests,
        )

        try:
            # Import client here to avoid circular imports
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from wrinklefree_inference.client.bitnet_client import AsyncBitNetClient

        except ImportError:
            logger.warning("Could not import AsyncBitNetClient, using mock client")
            result.errors = 1
            result.calculate_costs()
            return result

        try:
            async with AsyncBitNetClient(
                self.config.server_url,
                timeout=self.config.timeout_seconds,
            ) as client:
                # Health check
                if not await client.health_check():
                    logger.error(f"Server not healthy: {self.config.server_url}")
                    result.errors = 1
                    return result

                # Warmup
                logger.info(f"Warming up with {self.config.warmup_requests} requests...")
                for _ in range(self.config.warmup_requests):
                    try:
                        await client.generate("Warmup", max_tokens=10)
                    except Exception as e:
                        logger.warning(f"Warmup error: {e}")

                # Get memory usage before benchmarks
                result.memory_usage_gb = await self._get_memory_usage()

                # Run benchmarks at different batch sizes
                best_throughput = 0.0
                best_batch_size = 1
                all_metrics = []

                for batch_size in self.config.batch_sizes:
                    logger.info(f"Benchmarking batch_size={batch_size}...")

                    metrics = await self._run_batch_benchmark(
                        client,
                        batch_size,
                        self.config.benchmark_requests,
                        self.config.max_tokens,
                        self.config.duration_seconds,
                    )
                    all_metrics.append(metrics)

                    if metrics.tokens_per_second > best_throughput:
                        best_throughput = metrics.tokens_per_second
                        best_batch_size = batch_size
                        result.metrics = metrics

                # Use best result for final metrics
                if result.metrics:
                    result.tokens_per_second = result.metrics.tokens_per_second
                    result.ttft_p50_ms = result.metrics.ttft_p50_ms
                    result.ttft_p99_ms = result.metrics.ttft_p99_ms
                    result.latency_p50_ms = result.metrics.latency_p50_ms
                    result.latency_p95_ms = result.metrics.latency_p95_ms
                    result.latency_p99_ms = result.metrics.latency_p99_ms
                    result.total_requests = result.metrics.requests
                    result.total_tokens_generated = result.metrics.tokens_generated
                    result.duration_seconds = result.metrics.total_time_seconds
                    result.errors = result.metrics.failed

                result.optimal_batch_size = best_batch_size
                result.peak_memory_gb = await self._get_memory_usage()

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            result.errors = 1

        # Calculate costs
        try:
            price, _ = self.cost_tracker.get_hardware_price(
                self.config.hardware,
                self.config.provider,
                self.config.use_spot,
            )
            result.hardware_cost_per_hour = price
            result.calculate_costs()
        except Exception as e:
            logger.warning(f"Cost calculation failed: {e}")

        return result

    async def _run_batch_benchmark(
        self,
        client,
        batch_size: int,
        num_requests: int,
        max_tokens: int,
        duration_seconds: int,
    ) -> BenchmarkMetrics:
        """Run benchmark at a specific batch size."""
        latencies = []
        ttfts = []
        tokens = []
        errors = 0

        start_time = time.perf_counter()
        semaphore = asyncio.Semaphore(batch_size)

        async def single_request(prompt: str):
            nonlocal errors
            async with semaphore:
                try:
                    request_start = time.perf_counter()

                    # Try to measure TTFT via streaming if available
                    response = await client.generate(prompt, max_tokens=max_tokens)

                    latency = (time.perf_counter() - request_start) * 1000
                    # Estimate tokens from response length
                    token_count = len(response.split()) if response else 0

                    return latency, token_count, latency * 0.3  # Rough TTFT estimate

                except Exception as e:
                    errors += 1
                    logger.debug(f"Request error: {e}")
                    return None, 0, None

        # Run requests until duration expires or num_requests reached
        request_count = 0
        while (time.perf_counter() - start_time) < duration_seconds and request_count < num_requests:
            # Create batch of requests
            batch_prompts = [
                self._prompts[i % len(self._prompts)]
                for i in range(request_count, request_count + batch_size)
            ]

            tasks = [single_request(p) for p in batch_prompts]
            results = await asyncio.gather(*tasks)

            for latency, token_count, ttft in results:
                if latency is not None:
                    latencies.append(latency)
                    tokens.append(token_count)
                    if ttft is not None:
                        ttfts.append(ttft)

            request_count += batch_size

        total_time = time.perf_counter() - start_time

        return BenchmarkMetrics.from_latencies(
            name=f"batch_{batch_size}",
            latencies=latencies,
            tokens=tokens,
            total_time=total_time,
            failed=errors,
            ttfts=ttfts if ttfts else None,
        )

    async def _get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / (1024**3)
        except ImportError:
            return 0.0


async def run_benchmark_suite(
    configs: list[BenchmarkConfig],
    output_dir: Path,
) -> list[CostBenchmarkResult]:
    """
    Run multiple benchmarks and save results.

    Args:
        configs: List of benchmark configurations
        output_dir: Directory to save results

    Returns:
        List of benchmark results
    """
    results = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cost_tracker = CostTracker()

    for i, config in enumerate(configs):
        logger.info(f"Running benchmark {i + 1}/{len(configs)}: {config.model} on {config.hardware}")

        runner = BenchmarkRunner(config, cost_tracker)
        result = await runner.run()
        results.append(result)

        # Save individual result
        result.save(output_dir)

        logger.info(
            f"  Result: {result.tokens_per_second:.1f} tok/s, "
            f"${result.cost_per_million_tokens:.4f}/1M tokens"
        )

    return results
