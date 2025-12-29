"""Benchmark metrics dataclass and utilities."""

from dataclasses import dataclass, field
from typing import Optional
import json
from pathlib import Path


@dataclass
class BenchmarkMetrics:
    """Metrics collected during benchmark runs.

    Primary optimization target is convergence_per_sec_per_gb which measures
    learning efficiency as loss reduction per second normalized by memory.

    convergence_per_sec_per_gb = (initial_loss - final_loss) / wall_time / peak_memory

    This captures: how much learning happens per unit compute cost.
    """

    # Primary optimization target: convergence efficiency
    convergence_per_sec_per_gb: float

    # Secondary metric: raw throughput efficiency
    tokens_per_sec_per_gb: float

    # Component metrics
    throughput_tokens_per_sec: float
    peak_memory_gb: float
    allocated_memory_gb: float

    # Training quality metrics
    final_loss: float
    initial_loss: float
    loss_reduction_rate: float  # (initial_loss - final_loss) / steps
    grad_norm_mean: float
    grad_norm_std: float

    # Configuration metadata
    optimizer_type: str
    batch_size: int
    learning_rate: float
    gradient_accumulation_steps: int

    # Distillation metrics (Stage 3)
    lambda_logits: Optional[float] = None
    gamma_attention: Optional[float] = None
    temperature: Optional[float] = None

    # Influence function metrics
    influence_enabled: bool = False
    influence_lambda_reg: Optional[float] = None
    influence_threshold: Optional[float] = None
    weight_update_interval: Optional[int] = None

    # Run metadata
    num_steps: int = 0
    wall_time_seconds: float = 0.0
    trial_id: int = 0

    # Additional tracking
    extra_metrics: dict = field(default_factory=dict)

    @classmethod
    def compute(
        cls,
        throughput_tokens_per_sec: float,
        peak_memory_gb: float,
        allocated_memory_gb: float,
        final_loss: float,
        initial_loss: float,
        num_steps: int,
        grad_norms: list[float],
        optimizer_type: str,
        batch_size: int,
        learning_rate: float,
        gradient_accumulation_steps: int,
        wall_time_seconds: float,
        trial_id: int = 0,
        lambda_logits: Optional[float] = None,
        gamma_attention: Optional[float] = None,
        temperature: Optional[float] = None,
        influence_enabled: bool = False,
        influence_lambda_reg: Optional[float] = None,
        influence_threshold: Optional[float] = None,
        weight_update_interval: Optional[int] = None,
        extra_metrics: Optional[dict] = None,
    ) -> "BenchmarkMetrics":
        """Compute benchmark metrics from raw measurements.

        Args:
            throughput_tokens_per_sec: Measured token throughput
            peak_memory_gb: Peak GPU memory in GB
            allocated_memory_gb: Currently allocated memory in GB
            final_loss: Loss at end of benchmark
            initial_loss: Loss at start of benchmark
            num_steps: Number of training steps
            grad_norms: List of gradient norms from each step
            optimizer_type: Type of optimizer used
            batch_size: Batch size
            learning_rate: Learning rate
            gradient_accumulation_steps: Gradient accumulation steps
            wall_time_seconds: Total wall clock time
            trial_id: Trial identifier
            lambda_logits: Distillation lambda coefficient
            gamma_attention: Attention distillation coefficient
            temperature: Distillation temperature
            influence_enabled: Whether influence functions are enabled
            influence_lambda_reg: Influence regularization
            influence_threshold: Influence threshold
            weight_update_interval: Weight update interval
            extra_metrics: Additional metrics to track

        Returns:
            BenchmarkMetrics instance with computed values
        """
        import numpy as np

        # Compute throughput efficiency (secondary metric)
        tokens_per_sec_per_gb = throughput_tokens_per_sec / max(peak_memory_gb, 0.1)

        # Compute loss reduction rate (per step)
        loss_reduction_rate = (initial_loss - final_loss) / max(num_steps, 1)

        # Compute PRIMARY metric: convergence efficiency
        # = (loss reduction) / (wall time) / (memory)
        # This measures: how much learning per second per GB of memory
        loss_reduction = max(initial_loss - final_loss, 0.0)  # Ensure non-negative
        convergence_per_sec_per_gb = loss_reduction / max(wall_time_seconds, 0.1) / max(peak_memory_gb, 0.1)

        # Compute gradient norm statistics
        if grad_norms:
            grad_norm_mean = float(np.mean(grad_norms))
            grad_norm_std = float(np.std(grad_norms))
        else:
            grad_norm_mean = 0.0
            grad_norm_std = 0.0

        return cls(
            convergence_per_sec_per_gb=convergence_per_sec_per_gb,
            tokens_per_sec_per_gb=tokens_per_sec_per_gb,
            throughput_tokens_per_sec=throughput_tokens_per_sec,
            peak_memory_gb=peak_memory_gb,
            allocated_memory_gb=allocated_memory_gb,
            final_loss=final_loss,
            initial_loss=initial_loss,
            loss_reduction_rate=loss_reduction_rate,
            grad_norm_mean=grad_norm_mean,
            grad_norm_std=grad_norm_std,
            optimizer_type=optimizer_type,
            batch_size=batch_size,
            learning_rate=learning_rate,
            gradient_accumulation_steps=gradient_accumulation_steps,
            lambda_logits=lambda_logits,
            gamma_attention=gamma_attention,
            temperature=temperature,
            influence_enabled=influence_enabled,
            influence_lambda_reg=influence_lambda_reg,
            influence_threshold=influence_threshold,
            weight_update_interval=weight_update_interval,
            num_steps=num_steps,
            wall_time_seconds=wall_time_seconds,
            trial_id=trial_id,
            extra_metrics=extra_metrics or {},
        )

    def to_dict(self) -> dict:
        """Convert metrics to dictionary for Ax reporting."""
        return {
            "convergence_per_sec_per_gb": self.convergence_per_sec_per_gb,
            "tokens_per_sec_per_gb": self.tokens_per_sec_per_gb,
            "throughput_tokens_per_sec": self.throughput_tokens_per_sec,
            "peak_memory_gb": self.peak_memory_gb,
            "allocated_memory_gb": self.allocated_memory_gb,
            "final_loss": self.final_loss,
            "initial_loss": self.initial_loss,
            "loss_reduction_rate": self.loss_reduction_rate,
            "grad_norm_mean": self.grad_norm_mean,
            "grad_norm_std": self.grad_norm_std,
            "optimizer_type": self.optimizer_type,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "lambda_logits": self.lambda_logits,
            "gamma_attention": self.gamma_attention,
            "temperature": self.temperature,
            "influence_enabled": self.influence_enabled,
            "influence_lambda_reg": self.influence_lambda_reg,
            "influence_threshold": self.influence_threshold,
            "weight_update_interval": self.weight_update_interval,
            "num_steps": self.num_steps,
            "wall_time_seconds": self.wall_time_seconds,
            "trial_id": self.trial_id,
            **self.extra_metrics,
        }

    def to_ax_metrics(self) -> dict:
        """Convert to Ax-compatible metrics dictionary.

        Returns only the objective metric registered with the Ax experiment.
        The primary objective is convergence_per_sec_per_gb which measures
        learning efficiency (loss reduction per second per GB memory).
        """
        return {
            "convergence_per_sec_per_gb": (self.convergence_per_sec_per_gb, 0.0),
        }

    def save(self, path: Path) -> None:
        """Save metrics to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "BenchmarkMetrics":
        """Load metrics from JSON file."""
        with open(path) as f:
            data = json.load(f)

        # Extract extra_metrics
        known_fields = {
            "convergence_per_sec_per_gb", "tokens_per_sec_per_gb", "throughput_tokens_per_sec", "peak_memory_gb",
            "allocated_memory_gb", "final_loss", "initial_loss", "loss_reduction_rate",
            "grad_norm_mean", "grad_norm_std", "optimizer_type", "batch_size",
            "learning_rate", "gradient_accumulation_steps", "lambda_logits",
            "gamma_attention", "temperature", "influence_enabled", "influence_lambda_reg",
            "influence_threshold", "weight_update_interval", "num_steps",
            "wall_time_seconds", "trial_id", "extra_metrics",
        }
        extra = {k: v for k, v in data.items() if k not in known_fields}

        return cls(
            convergence_per_sec_per_gb=data.get("convergence_per_sec_per_gb", 0.0),
            tokens_per_sec_per_gb=data["tokens_per_sec_per_gb"],
            throughput_tokens_per_sec=data["throughput_tokens_per_sec"],
            peak_memory_gb=data["peak_memory_gb"],
            allocated_memory_gb=data["allocated_memory_gb"],
            final_loss=data["final_loss"],
            initial_loss=data["initial_loss"],
            loss_reduction_rate=data["loss_reduction_rate"],
            grad_norm_mean=data["grad_norm_mean"],
            grad_norm_std=data["grad_norm_std"],
            optimizer_type=data["optimizer_type"],
            batch_size=data["batch_size"],
            learning_rate=data["learning_rate"],
            gradient_accumulation_steps=data["gradient_accumulation_steps"],
            lambda_logits=data.get("lambda_logits"),
            gamma_attention=data.get("gamma_attention"),
            temperature=data.get("temperature"),
            influence_enabled=data.get("influence_enabled", False),
            influence_lambda_reg=data.get("influence_lambda_reg"),
            influence_threshold=data.get("influence_threshold"),
            weight_update_interval=data.get("weight_update_interval"),
            num_steps=data.get("num_steps", 0),
            wall_time_seconds=data.get("wall_time_seconds", 0.0),
            trial_id=data.get("trial_id", 0),
            extra_metrics=extra or data.get("extra_metrics", {}),
        )

    def __str__(self) -> str:
        """Human-readable summary."""
        lines = [
            f"=== Benchmark Metrics (Trial {self.trial_id}) ===",
            f"Convergence Efficiency: {self.convergence_per_sec_per_gb:.4f} loss/sec/GB",
            f"Throughput Efficiency: {self.tokens_per_sec_per_gb:.1f} tokens/sec/GB",
            f"Loss: {self.initial_loss:.4f} → {self.final_loss:.4f} (Δ={self.initial_loss - self.final_loss:.4f})",
            f"Peak Memory: {self.peak_memory_gb:.2f} GB",
            f"Optimizer: {self.optimizer_type}",
            f"Batch Size: {self.batch_size}",
            f"Learning Rate: {self.learning_rate:.2e}",
        ]
        if self.influence_enabled:
            lines.append(f"Influence: enabled (lambda={self.influence_lambda_reg})")
        return "\n".join(lines)
