#!/usr/bin/env python3
"""Ax Bayesian optimization for Stage 2 (pre-training) hyperparameters.

Usage:
    uv run python scripts/benchmark_stage2.py --num-trials 10
    uv run python scripts/benchmark_stage2.py --num-trials 20 --model smollm2_135m
"""

import argparse
import logging
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.core.runner import BenchmarkRunner, RunnerConfig
from benchmark.optimization.ax_client import BenchmarkAxClient, load_search_space

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_optimization(
    num_trials: int = 10,
    model_name: str = "HuggingFaceTB/SmolLM2-135M",
    stage1_checkpoint: Path | None = None,
    output_dir: Path = Path("./benchmark_results/stage2"),
    measurement_steps: int = 300,
    warmup_steps: int = 20,
) -> None:
    """Run Ax optimization for Stage 2 training hyperparameters.

    Args:
        num_trials: Number of optimization trials to run
        model_name: HuggingFace model name
        stage1_checkpoint: Path to Stage 1 checkpoint (optional)
        output_dir: Directory for results
        measurement_steps: Steps per trial for measurement
        warmup_steps: Warmup steps before measurement
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load search space
    search_space_path = Path(__file__).parent.parent / "benchmark/config/search_space.yaml"
    if search_space_path.exists():
        search_space = load_search_space(search_space_path)
    else:
        from benchmark.optimization.ax_client import create_default_search_space
        search_space = create_default_search_space()

    # Create Ax client
    ax_client = BenchmarkAxClient(
        search_space_config=search_space,
        experiment_name=f"stage2_{model_name.split('/')[-1]}",
    )

    # Create runner
    runner_config = RunnerConfig(
        warmup_steps=warmup_steps,
        measurement_steps=measurement_steps,
        sequence_length=512,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    runner = BenchmarkRunner(
        runner_config=runner_config,
        model_name=model_name,
        stage1_checkpoint=stage1_checkpoint,
    )

    # Run optimization loop
    logger.info(f"Starting {num_trials} trials for Stage 2 optimization")

    for i in range(num_trials):
        try:
            params, trial_idx = ax_client.get_next_trial()
            logger.info(f"Trial {trial_idx}/{num_trials}: {params}")

            metrics = runner.run_trial(params, trial_id=trial_idx)
            ax_client.complete_trial(trial_idx, metrics)

            # Save experiment state after each trial
            ax_client.save_experiment(output_dir / "experiment.json")

        except Exception as e:
            logger.error(f"Trial {i} failed: {e}")
            ax_client.mark_trial_failed(trial_idx, str(e))

    # Report best parameters
    best_params = ax_client.get_best_parameters()
    logger.info(f"Best parameters: {best_params}")

    # Save final results
    df = ax_client.get_trials_dataframe()
    df.to_csv(output_dir / "trials.csv", index=False)
    logger.info(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Ax optimization for Stage 2 training")
    parser.add_argument("--num-trials", type=int, default=10, help="Number of optimization trials")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-135M", help="Model name")
    parser.add_argument("--stage1-checkpoint", type=Path, default=None, help="Stage 1 checkpoint path")
    parser.add_argument("--output-dir", type=Path, default=Path("./benchmark_results/stage2"))
    parser.add_argument("--measurement-steps", type=int, default=300, help="Steps per trial")
    parser.add_argument("--warmup-steps", type=int, default=20, help="Warmup steps")
    args = parser.parse_args()

    run_optimization(
        num_trials=args.num_trials,
        model_name=args.model,
        stage1_checkpoint=args.stage1_checkpoint,
        output_dir=args.output_dir,
        measurement_steps=args.measurement_steps,
        warmup_steps=args.warmup_steps,
    )


if __name__ == "__main__":
    main()
