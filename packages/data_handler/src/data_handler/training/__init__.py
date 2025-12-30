"""Training infrastructure - active components.

Legacy components (Trainer, TrainingStage, schedulers) have been moved to
cheapertraining._legacy.training
"""

from data_handler.training.early_stopping import PlateauEarlyStopping
from data_handler.training.gradient_clipping import (
    ZClip,
    ZClipStats,
    clip_grad_with_zclip,
)
from data_handler.training.optimizer import (
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
