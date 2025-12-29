"""Training infrastructure - active components.

Legacy components (Trainer, TrainingStage, schedulers) have been moved to
cheapertraining._legacy.training
"""

from cheapertraining.training.early_stopping import PlateauEarlyStopping
from cheapertraining.training.optimizer import (
    InfluenceAwareOptimizer,
    create_optimizer,
    get_parameter_groups,
    get_num_parameters,
)

__all__ = [
    "InfluenceAwareOptimizer",
    "PlateauEarlyStopping",
    "create_optimizer",
    "get_parameter_groups",
    "get_num_parameters",
]
