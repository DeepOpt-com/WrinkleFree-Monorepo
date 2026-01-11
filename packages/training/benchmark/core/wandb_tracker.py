"""W&B tracking utilities with intelligent naming.

Provides unified tracking for:
- Training runs (stage 1, 1.9, 2, 3)
- Inference benchmarks (throughput, latency)
- MoE experiments (routing analysis)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not installed, tracking disabled")


@dataclass
class InferenceMetrics:
    """Metrics for inference benchmarks."""

    # Performance metrics
    tokens_per_sec_prompt: float
    tokens_per_sec_gen: float
    latency_first_token_ms: float
    latency_per_token_ms: float

    # Resource metrics
    peak_memory_gb: float
    num_threads: int
    batch_size: int

    # Configuration
    model_name: str
    quant_type: str
    context_size: int

    # MoE-specific (optional)
    num_experts: int = 0
    top_k: int = 0
    router_type: str = ""

    # Extra metrics
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        d = {
            "inference/tokens_per_sec_prompt": self.tokens_per_sec_prompt,
            "inference/tokens_per_sec_gen": self.tokens_per_sec_gen,
            "inference/latency_first_token_ms": self.latency_first_token_ms,
            "inference/latency_per_token_ms": self.latency_per_token_ms,
            "inference/peak_memory_gb": self.peak_memory_gb,
            "config/num_threads": self.num_threads,
            "config/batch_size": self.batch_size,
            "config/model_name": self.model_name,
            "config/quant_type": self.quant_type,
            "config/context_size": self.context_size,
        }
        if self.num_experts > 0:
            d["moe/num_experts"] = self.num_experts
            d["moe/top_k"] = self.top_k
            d["moe/router_type"] = self.router_type
        d.update(self.extra)
        return d


class WandBTracker:
    """Unified W&B tracking with intelligent naming.

    Handles both training and inference benchmark logging with
    consistent naming conventions.

    Usage:
        # Training
        tracker = WandBTracker(project="wrinklefree")
        tracker.init_training_run(config)
        tracker.log_training_step({"loss": 0.5, "lr": 1e-4})
        tracker.finish()

        # Inference benchmark
        tracker = WandBTracker(project="wrinklefree-inference")
        tracker.init_benchmark_run(model_name="bitnet-2b", quant_type="i2_s")
        tracker.log_inference_metrics(metrics)
        tracker.finish()
    """

    def __init__(
        self,
        project: str = "wrinklefree",
        entity: Optional[str] = None,
        enabled: bool = True,
        tags: Optional[list[str]] = None,
    ):
        self.project = project
        self.entity = entity
        self.enabled = enabled and WANDB_AVAILABLE
        self.tags = tags or []
        self.run = None

    def init_training_run(
        self,
        config: Any,
        name: Optional[str] = None,
    ) -> Optional["wandb.Run"]:
        """Initialize a training run with intelligent naming.

        Args:
            config: Hydra/OmegaConf config or dict
            name: Optional explicit name (otherwise auto-generated)

        Returns:
            W&B run object or None if disabled
        """
        if not self.enabled:
            return None

        # Auto-generate name if not provided
        if name is None:
            from wf_train.training.run_naming import generate_run_name
            name = generate_run_name(config)

        # Convert OmegaConf to dict if needed
        if hasattr(config, "to_container"):
            from omegaconf import OmegaConf
            config_dict = OmegaConf.to_container(config, resolve=True)
        elif hasattr(config, "to_dict"):
            config_dict = config.to_dict()
        else:
            config_dict = dict(config) if config else {}

        self.run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=name,
            config=config_dict,
            tags=self.tags,
            reinit=True,
        )
        logger.info(f"W&B run initialized: {name}")
        return self.run

    def init_benchmark_run(
        self,
        model_name: str,
        quant_type: str = "i2_s",
        context_size: int = 4096,
        num_threads: int = 0,
        batch_size: int = 1,
        num_experts: int = 0,
        top_k: int = 0,
        name: Optional[str] = None,
    ) -> Optional["wandb.Run"]:
        """Initialize an inference benchmark run.

        Args:
            model_name: Model identifier
            quant_type: Quantization type
            context_size: Context window size
            num_threads: Number of inference threads
            batch_size: Batch size
            num_experts: Number of MoE experts (0 for dense)
            top_k: MoE top-k routing
            name: Optional explicit name

        Returns:
            W&B run object or None if disabled
        """
        if not self.enabled:
            return None

        # Auto-generate name if not provided
        if name is None:
            if num_experts > 0:
                from wf_train.training.run_naming import generate_moe_benchmark_name
                name = generate_moe_benchmark_name(
                    model_name, num_experts, top_k, quant_type, context_size, num_threads
                )
            else:
                from wf_train.training.run_naming import generate_benchmark_name
                name = generate_benchmark_name(
                    model_name, quant_type, context_size, num_threads, batch_size
                )

        config = {
            "model_name": model_name,
            "quant_type": quant_type,
            "context_size": context_size,
            "num_threads": num_threads,
            "batch_size": batch_size,
            "num_experts": num_experts,
            "top_k": top_k,
        }

        tags = self.tags + ["inference", "benchmark"]
        if num_experts > 0:
            tags.append("moe")

        self.run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=name,
            config=config,
            tags=tags,
            reinit=True,
        )
        logger.info(f"W&B benchmark run initialized: {name}")
        return self.run

    def log_training_step(
        self,
        metrics: dict[str, Any],
        step: Optional[int] = None,
    ) -> None:
        """Log training metrics.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        if not self.enabled or self.run is None:
            return
        self.run.log(metrics, step=step)

    def log_inference_metrics(
        self,
        metrics: InferenceMetrics,
        step: Optional[int] = None,
    ) -> None:
        """Log inference benchmark metrics.

        Args:
            metrics: InferenceMetrics dataclass
            step: Optional step number
        """
        if not self.enabled or self.run is None:
            return
        self.run.log(metrics.to_dict(), step=step)

    def log_summary(self, metrics: dict[str, Any]) -> None:
        """Log summary metrics at end of run.

        Args:
            metrics: Dictionary of summary metrics
        """
        if not self.enabled or self.run is None:
            return
        for key, value in metrics.items():
            self.run.summary[key] = value

    def finish(self) -> None:
        """Finish the W&B run."""
        if self.run is not None:
            self.run.finish()
            self.run = None
            logger.info("W&B run finished")


def create_inference_tracker(
    model_name: str,
    quant_type: str = "i2_s",
    context_size: int = 4096,
    num_threads: int = 0,
    project: str = "wrinklefree-inference",
    entity: Optional[str] = None,
) -> WandBTracker:
    """Convenience function to create and initialize an inference tracker.

    Args:
        model_name: Model identifier
        quant_type: Quantization type
        context_size: Context window size
        num_threads: Number of inference threads
        project: W&B project name
        entity: W&B entity (team/user)

    Returns:
        Initialized WandBTracker
    """
    tracker = WandBTracker(project=project, entity=entity, tags=["inference"])
    tracker.init_benchmark_run(
        model_name=model_name,
        quant_type=quant_type,
        context_size=context_size,
        num_threads=num_threads,
    )
    return tracker


def log_benchmark_results(
    metrics: "BenchmarkMetrics",
    project: str = "wrinklefree-training",
    entity: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> None:
    """One-shot logging of benchmark results to W&B.

    Args:
        metrics: BenchmarkMetrics from benchmark runner
        project: W&B project name
        entity: W&B entity
        tags: Additional tags
    """
    if not WANDB_AVAILABLE:
        logger.warning("wandb not available, skipping logging")
        return

    from wf_train.training.run_naming import _generate_suffix

    # Generate a name based on the trial
    suffix = _generate_suffix(3)
    name = f"trial{metrics.trial_id}-{metrics.optimizer_type}-{suffix}"

    run = wandb.init(
        project=project,
        entity=entity,
        name=name,
        config=metrics.to_dict(),
        tags=tags or ["benchmark", "training"],
        reinit=True,
    )

    # Log all metrics
    run.log(metrics.to_dict())

    # Log summary
    run.summary["convergence_per_sec_per_gb"] = metrics.convergence_per_sec_per_gb
    run.summary["tokens_per_sec_per_gb"] = metrics.tokens_per_sec_per_gb
    run.summary["final_loss"] = metrics.final_loss

    run.finish()
    logger.info(f"Logged benchmark results to W&B: {name}")
