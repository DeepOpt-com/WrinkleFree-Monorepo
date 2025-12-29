#!/usr/bin/env python3
"""Standalone evaluation script.

Usage:
    python scripts/evaluate.py --model-path /path/to/model --benchmark bitdistill
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wrinklefree_eval.api import evaluate, list_benchmarks


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a model on BitDistill benchmarks"
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--benchmark",
        default="bitdistill",
        choices=["bitdistill", "glue", "summarization", "smoke_test"],
        help="Benchmark preset to run",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run on (cuda, cpu)",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype",
    )
    parser.add_argument(
        "--batch-size",
        default="auto",
        help="Batch size (auto for automatic)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit samples per task",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run smoke test (10 samples per task)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save results",
    )
    parser.add_argument(
        "--list-benchmarks",
        action="store_true",
        help="List available benchmarks and exit",
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="W&B project name (enables logging)",
    )
    parser.add_argument(
        "--wandb-run-id",
        default=None,
        help="W&B run ID for resuming",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="W&B run display name",
    )

    args = parser.parse_args()

    if args.list_benchmarks:
        benchmarks = list_benchmarks()
        print("\nAvailable Benchmarks:")
        for name, tasks in benchmarks.items():
            print(f"\n  {name}:")
            for task in tasks:
                print(f"    - {task}")
        return

    # Convert batch_size to int if not "auto"
    batch_size = args.batch_size
    if batch_size != "auto":
        batch_size = int(batch_size)

    print(f"\nEvaluating: {args.model_path}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Device: {args.device}, Dtype: {args.dtype}")
    if args.smoke_test:
        print("Mode: Smoke test (10 samples)")
    print()

    results = evaluate(
        model_path=args.model_path,
        benchmark=args.benchmark,
        device=args.device,
        dtype=args.dtype,
        batch_size=batch_size,
        limit=args.limit,
        smoke_test=args.smoke_test,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        wandb_run_id=args.wandb_run_id,
        wandb_run_name=args.wandb_run_name,
    )

    # Print results
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)

    for task_name, task_results in results.items():
        print(f"\n{task_name}:")
        for metric, value in task_results.items():
            if "_stderr" not in metric:
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")

    if args.output_dir:
        print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
