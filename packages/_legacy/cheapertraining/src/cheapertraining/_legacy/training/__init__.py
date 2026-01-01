"""Legacy training components."""

from cheapertraining._legacy.training.trainer import Trainer
from cheapertraining._legacy.training.stages import TrainingStage, StageConfig
from cheapertraining._legacy.training.scheduler import (
    create_scheduler,
    LinearWarmupLinearDecay,
    LinearWarmupLinearDecayToZero,
    CosineWarmup,
)

__all__ = [
    "Trainer",
    "TrainingStage",
    "StageConfig",
    "create_scheduler",
    "LinearWarmupLinearDecay",
    "LinearWarmupLinearDecayToZero",
    "CosineWarmup",
]
