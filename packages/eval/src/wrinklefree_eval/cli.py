"""CLI interface for WrinkleFree-Eval.

Usage:
    # With Hydra config
    uv run python -m wrinklefree_eval model_path=/path/to/model benchmark=bitdistill

    # Direct CLI
    wrinklefree-eval --model-path /path/to/model --benchmark bitdistill
"""

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.table import Table

from wrinklefree_eval.api import evaluate, list_benchmarks

console = Console()
logger = logging.getLogger(__name__)


def print_results(results: dict, benchmark: str):
    """Pretty-print evaluation results using Rich."""
    console.print(f"\n[bold green]Evaluation Results: {benchmark}[/bold green]\n")

    for task_name, task_results in results.items():
        table = Table(title=task_name, show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right")

        for metric, value in task_results.items():
            if "_stderr" in metric:
                continue  # Skip stderr, show with main metric

            # Format value
            if isinstance(value, float):
                formatted = f"{value:.4f}"
                # Add stderr if available
                stderr_key = f"{metric}_stderr"
                if stderr_key in task_results:
                    stderr = task_results[stderr_key]
                    formatted += f" (±{stderr:.4f})"
            else:
                formatted = str(value)

            table.add_row(metric, formatted)

        console.print(table)
        console.print()


@hydra.main(version_base=None, config_path="../../../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    """Main entry point with Hydra configuration."""
    # Print config
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Validate required fields
    if cfg.model_path is None:
        console.print("[red]Error: model_path is required[/red]")
        console.print("Usage: python -m wrinklefree_eval model_path=/path/to/model")
        sys.exit(1)

    # Get benchmark config
    benchmark_name = cfg.benchmark.get("name", "bitdistill")
    tasks = cfg.benchmark.get("tasks", None)
    num_fewshot = cfg.benchmark.get("num_fewshot", None)

    # Get limits
    limits = cfg.benchmark.get("limits", {})
    # Use first non-null limit as global limit, or None
    limit = None
    for task_limit in limits.values():
        if task_limit is not None:
            limit = task_limit
            break

    # Override with smoke_test if enabled
    if cfg.get("smoke_test", False):
        limit = cfg.get("smoke_test_limit", 10)
        console.print(f"[yellow]Smoke test mode: {limit} samples per task[/yellow]")

    # Run evaluation
    console.print(f"[bold]Evaluating model:[/bold] {cfg.model_path}")
    console.print(f"[bold]Benchmark:[/bold] {benchmark_name}")
    console.print(f"[bold]Device:[/bold] {cfg.device}, [bold]Dtype:[/bold] {cfg.dtype}")
    console.print()

    try:
        results = evaluate(
            model_path=cfg.model_path,
            benchmark=benchmark_name,
            tasks=tasks,
            device=cfg.device,
            dtype=cfg.dtype,
            batch_size=cfg.batch_size,
            num_fewshot=num_fewshot,
            limit=limit,
            smoke_test=cfg.get("smoke_test", False),
            output_dir=cfg.output_dir if cfg.save_results else None,
            trust_remote_code=cfg.model.get("trust_remote_code", True),
            verbosity=cfg.verbosity,
        )

        # Print results
        print_results(results, benchmark_name)

        # Save summary
        if cfg.save_results:
            console.print(f"[green]Results saved to {cfg.output_dir}[/green]")

    except Exception as e:
        console.print(f"[red]Evaluation failed: {e}[/red]")
        logger.exception("Evaluation error")
        sys.exit(1)


def cli_list_benchmarks():
    """CLI command to list available benchmarks."""
    console.print("\n[bold]Available Benchmarks:[/bold]\n")

    benchmarks = list_benchmarks()
    for name, tasks in benchmarks.items():
        console.print(f"  [cyan]{name}[/cyan]")
        for task in tasks:
            console.print(f"    - {task}")
        console.print()


if __name__ == "__main__":
    main()
