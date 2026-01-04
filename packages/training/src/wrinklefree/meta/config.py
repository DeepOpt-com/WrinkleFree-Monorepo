"""Configuration for meta-optimization outer loop.

References:
- LibMOON (NeurIPS 2024): https://arxiv.org/abs/2409.02969
- ScaleBiO (2024): https://arxiv.org/abs/2406.19976
- DataInf (ICLR 2024): https://openreview.net/forum?id=9m02ib92Wz
"""

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class ParetoConfig:
    """Configuration for Pareto gradient solver.

    Reference: LibMOON (https://github.com/xzhang2523/libmoon)
    """

    method: Literal["mgda", "epo", "linear"] = "mgda"
    max_iter: int = 10
    tol: float = 1e-4
    normalize_gradients: bool = True
    # For EPO: preference weights for each validation objective
    preferences: Optional[list[float]] = None


@dataclass
class ValidationObjectiveConfig:
    """Configuration for a validation objective in Pareto optimization."""

    name: str
    loader: str  # Reference to a dataloader config
    weight: float = 1.0  # Initial preference weight


@dataclass
class MetaConstraintsConfig:
    """Constraints for meta-parameters."""

    dataset_weight_range: tuple[float, float] = (0.05, 0.60)
    objective_weight_range: tuple[float, float] = (0.01, 2.0)
    lr_scale_range: tuple[float, float] = (0.5, 2.0)


@dataclass
class MetaGradientConfig:
    """Configuration for meta-gradient estimation."""

    method: Literal["datainf", "finite_difference"] = "datainf"
    lambda_reg: float = 1e-4
    samples_per_source: int = 256
    use_aggregated_gradient: bool = True


@dataclass
class MetaOptimizationConfig:
    """Configuration for meta-optimization outer loop.

    This enables joint optimization of:
    - Dataset mixture weights (extends existing InfluenceTracker)
    - Objective weights (CE, DLM, distillation, etc.)
    - Learning rate scales (per parameter group)

    Using influence-based gradient estimation with multi-objective
    Pareto optimization for validation signals.
    """

    enabled: bool = False

    # Which meta-parameters to optimize
    optimize_dataset_weights: bool = True
    optimize_objective_weights: bool = True
    optimize_learning_rates: bool = False  # Experimental

    # Update schedule
    update_interval: int = 1000
    warmup_steps: int = 500

    # Gradient estimation
    gradient: MetaGradientConfig = field(default_factory=MetaGradientConfig)

    # Pareto optimization
    pareto: ParetoConfig = field(default_factory=ParetoConfig)

    # Validation objectives for multi-objective optimization
    validation_objectives: list[ValidationObjectiveConfig] = field(default_factory=list)

    # Meta-learning hyperparameters
    meta_lr: float = 0.1
    meta_momentum: float = 0.9

    # Constraints
    constraints: MetaConstraintsConfig = field(default_factory=MetaConstraintsConfig)

    # Logging
    log_interval: int = 100
    log_pareto_front: bool = True

    def __post_init__(self):
        """Set default validation objectives if none provided."""
        if self.enabled and not self.validation_objectives:
            # Default to validation perplexity as single objective
            self.validation_objectives = [
                ValidationObjectiveConfig(
                    name="validation_perplexity",
                    loader="validation",
                    weight=1.0,
                )
            ]
