"""Training infrastructure - active components.

Legacy components (Trainer, TrainingStage, schedulers) have been moved to
cheapertraining._legacy.training
"""

from cheapertraining.training.early_stopping import PlateauEarlyStopping
from cheapertraining.training.gradient_clipping import (
    ZClip,
    ZClipStats,
    clip_grad_with_zclip,
)
from cheapertraining.training.optimizer import (
    InfluenceAwareOptimizer,
    create_optimizer,
    get_parameter_groups,
    get_num_parameters,
)

__all__ = [
    # Gradient clipping
    "ZClip",
    "ZClipStats",
    "clip_grad_with_zclip",
    # Early stopping
    "PlateauEarlyStopping",
    # Optimizer utilities
    "InfluenceAwareOptimizer",
    "create_optimizer",
    "get_parameter_groups",
    "get_num_parameters",
]
