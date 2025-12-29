"""Clean Python API for WrinkleFree evaluation.

Simple one-liner interface:
    from wrinklefree_eval import evaluate
    results = evaluate("path/to/model", benchmark="bitdistill")

With optional W&B logging:
    results = evaluate("path/to/model", wandb_project="my-project")
"""

from pathlib import Path
from typing import Any
import json
import logging
import os

import lm_eval
from lm_eval import evaluator
from lm_eval.tasks import TaskManager
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# Task name mapping from our configs to lm-eval task names
# Note: lm-eval renamed glue_* tasks to just the task name in recent versions
TASK_MAPPING = {
    # GLUE tasks (built into lm-eval)
    "mnli": "mnli",
    "qnli": "qnli",
    "sst2": "sst2",
    # Summarization (use built-in cnn_dailymail)
    "cnn_dailymail_summarization": "cnn_dailymail",
}

# Benchmark presets
# Note: cnn_dailymail requires 'unitxt' package for generation tasks
BENCHMARK_PRESETS = {
    "bitdistill": ["mnli", "qnli", "sst2"],  # GLUE subset from BitDistill paper
    "glue": ["mnli", "qnli", "sst2"],
    "smoke_test": ["sst2"],  # Fast single-task validation
}


def list_benchmarks() -> dict[str, list[str]]:
    """List available benchmark presets and their tasks.

    Returns:
        Dict mapping benchmark name to list of task names
    """
    return BENCHMARK_PRESETS.copy()


def evaluate(
    model_path: str,
    benchmark: str = "bitdistill",
    tasks: list[str] | None = None,
    device: str = "cuda",
    dtype: str = "bfloat16",
    batch_size: int | str = "auto",
    num_fewshot: int | dict[str, int] | None = None,
    limit: int | None = None,
    smoke_test: bool = False,
    output_dir: str | None = None,
    use_bitnet: bool = False,
    trust_remote_code: bool = True,
    verbosity: str = "INFO",
    wandb_project: str | None = None,
    wandb_run_id: str | None = None,
    wandb_run_name: str | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Evaluate a model on BitDistill benchmarks.

    Simple one-liner API for evaluation:
        results = evaluate("path/to/model", benchmark="bitdistill")

    With W&B logging:
        results = evaluate("path/to/model", wandb_project="my-project")

    Args:
        model_path: HuggingFace model ID or local path to model
        benchmark: Benchmark preset ("bitdistill", "glue", "summarization", "smoke_test")
        tasks: Override tasks list (if None, uses benchmark preset)
        device: Device to run on ("cuda", "cpu")
        dtype: Model dtype ("float16", "bfloat16", "float32")
        batch_size: Batch size for evaluation ("auto" for automatic)
        num_fewshot: Number of few-shot examples (int for all tasks, or dict per task)
        limit: Limit number of samples per task (None = full dataset)
        smoke_test: Enable smoke test mode (limit=10 per task)
        output_dir: Directory to save results (None = don't save)
        use_bitnet: Use BitNet kernels if available
        trust_remote_code: Trust remote code in model config
        verbosity: Logging verbosity ("DEBUG", "INFO", "WARNING")
        wandb_project: W&B project name (None = no logging)
        wandb_run_id: W&B run ID for resuming (None = auto-generate)
        wandb_run_name: W&B run display name (None = auto-generate)
        **kwargs: Additional arguments passed to lm_eval.simple_evaluate

    Returns:
        Dict with results for each task:
        {
            "glue_sst2": {"accuracy": 0.92, ...},
            "cnn_dailymail_summarization": {"rouge1": 0.45, "rouge2": 0.21, ...},
            ...
        }
    """
    # Configure logging
    logging.basicConfig(level=getattr(logging, verbosity))

    # Initialize W&B if requested
    wandb_run = None
    if wandb_project:
        if not WANDB_AVAILABLE:
            logger.warning("wandb_project specified but wandb not installed. Install with: pip install wandb")
        else:
            wandb_run = wandb.init(
                project=wandb_project,
                id=wandb_run_id,
                name=wandb_run_name or f"eval-{benchmark}",
                config={
                    "model_path": model_path,
                    "benchmark": benchmark,
                    "device": device,
                    "dtype": dtype,
                    "batch_size": batch_size,
                    "limit": limit,
                    "smoke_test": smoke_test,
                },
                resume="allow" if wandb_run_id else None,
            )
            logger.info(f"W&B run initialized: {wandb_run.url}")

    # Resolve tasks from benchmark preset
    if tasks is None:
        if benchmark not in BENCHMARK_PRESETS:
            available = ", ".join(BENCHMARK_PRESETS.keys())
            raise ValueError(f"Unknown benchmark: {benchmark}. Available: {available}")
        tasks = BENCHMARK_PRESETS[benchmark]

    # Apply smoke test limits
    if smoke_test:
        limit = limit or 10
        logger.info(f"Smoke test mode: limiting to {limit} samples per task")

    # Map task names to lm-eval task names
    lm_eval_tasks = []
    for task in tasks:
        if task in TASK_MAPPING:
            lm_eval_tasks.append(TASK_MAPPING[task])
        else:
            # Assume it's already an lm-eval task name
            lm_eval_tasks.append(task)

    # Register custom tasks using TaskManager
    # Always create a TaskManager to ensure default tasks are available
    from lm_eval.tasks import TaskManager
    custom_tasks_dir = Path(__file__).parent / "tasks"
    if custom_tasks_dir.exists():
        # Include custom tasks in addition to defaults
        task_manager = TaskManager(include_path=str(custom_tasks_dir), include_defaults=True)
        logger.debug(f"Registered custom tasks from {custom_tasks_dir}")
    else:
        task_manager = TaskManager()

    # Build model arguments
    model_args = f"pretrained={model_path}"
    model_args += f",dtype={dtype}"
    model_args += f",trust_remote_code={trust_remote_code}"

    # Handle few-shot configuration
    if isinstance(num_fewshot, dict):
        # Per-task few-shot (not directly supported, use default)
        num_fewshot_arg = 0
    else:
        num_fewshot_arg = num_fewshot if num_fewshot is not None else 0

    logger.info(f"Evaluating {model_path} on tasks: {lm_eval_tasks}")
    logger.info(f"Device: {device}, Dtype: {dtype}, Batch size: {batch_size}")

    # Run evaluation using lm_eval
    results = evaluator.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=lm_eval_tasks,
        num_fewshot=num_fewshot_arg,
        batch_size=batch_size,
        device=device,
        limit=limit,
        task_manager=task_manager,
        **kwargs,
    )

    # Extract and format results
    formatted_results = _format_results(results)

    # Save results if output_dir specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results_file = output_path / "results.json"
        with open(results_file, "w") as f:
            json.dump(formatted_results, f, indent=2)
        logger.info(f"Results saved to {results_file}")

        # Save full results with samples
        full_results_file = output_path / "results_full.json"
        with open(full_results_file, "w") as f:
            # Convert to serializable format
            serializable = _make_serializable(results)
            json.dump(serializable, f, indent=2)

    # Log to W&B if initialized
    if wandb_run is not None:
        _log_to_wandb(formatted_results, wandb_run)
        wandb_run.finish()
        logger.info("W&B run finished")

    return formatted_results


def _log_to_wandb(results: dict[str, Any], run) -> None:
    """Log evaluation results to Weights & Biases.

    Args:
        results: Formatted evaluation results
        run: Active wandb run
    """
    # Log each metric with task prefix
    for task_name, task_results in results.items():
        for metric_name, value in task_results.items():
            if isinstance(value, (int, float)) and "_stderr" not in metric_name:
                run.log({f"{task_name}/{metric_name}": value})

    # Create summary table
    summary_data = []
    for task_name, task_results in results.items():
        row = {"task": task_name}
        for metric_name, value in task_results.items():
            if isinstance(value, (int, float)) and "_stderr" not in metric_name:
                row[metric_name] = value
        summary_data.append(row)

    if summary_data:
        table = wandb.Table(
            columns=list(summary_data[0].keys()),
            data=[list(row.values()) for row in summary_data],
        )
        run.log({"results_summary": table})


def _format_results(results: dict) -> dict[str, Any]:
    """Format lm_eval results into a cleaner structure.

    Args:
        results: Raw results from lm_eval.simple_evaluate

    Returns:
        Formatted results dict
    """
    formatted = {}

    if "results" not in results:
        return formatted

    for task_name, task_results in results["results"].items():
        formatted[task_name] = {}

        for metric_name, value in task_results.items():
            # Skip internal metrics
            if metric_name.startswith("_"):
                continue

            # Clean up metric names
            clean_name = metric_name
            if "," in metric_name:
                # Handle metrics like "acc,none"
                clean_name = metric_name.split(",")[0]

            # Handle stderr separately
            if "_stderr" in metric_name:
                continue

            formatted[task_name][clean_name] = value

            # Add stderr if available
            stderr_key = f"{metric_name}_stderr"
            if stderr_key in task_results:
                formatted[task_name][f"{clean_name}_stderr"] = task_results[stderr_key]

    return formatted


def _make_serializable(obj: Any) -> Any:
    """Convert object to JSON-serializable format."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif hasattr(obj, "item"):  # numpy/torch scalar
        return obj.item()
    else:
        return str(obj)


def evaluate_from_config(config_path: str) -> dict[str, Any]:
    """Evaluate using a Hydra config file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Evaluation results
    """
    cfg = OmegaConf.load(config_path)

    return evaluate(
        model_path=cfg.model_path,
        benchmark=cfg.get("benchmark", {}).get("name", "bitdistill"),
        device=cfg.get("device", "cuda"),
        dtype=cfg.get("dtype", "bfloat16"),
        batch_size=cfg.get("batch_size", "auto"),
        limit=cfg.get("benchmark", {}).get("limits", {}).get("default"),
        smoke_test=cfg.get("smoke_test", False),
        output_dir=cfg.get("output_dir"),
    )
