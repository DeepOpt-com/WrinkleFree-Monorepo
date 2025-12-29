"""Legacy CheaperTraining components.

These modules are no longer actively maintained and have been moved to _legacy.
For new projects, use only the influence and data mixing components from the main module.

Usage:
    from cheapertraining._legacy.models import MobileLLM, MobileLLMConfig
    from cheapertraining._legacy.training import Trainer, TrainingStage
"""

from cheapertraining._legacy.models import (
    MobileLLM,
    MobileLLMConfig,
    MobileLLM140MConfig,
    MobileLLM360MConfig,
    MobileLLM950MConfig,
)
from cheapertraining._legacy.training.trainer import Trainer
from cheapertraining._legacy.training.stages import TrainingStage, StageConfig

__all__ = [
    "MobileLLM",
    "MobileLLMConfig",
    "MobileLLM140MConfig",
    "MobileLLM360MConfig",
    "MobileLLM950MConfig",
    "Trainer",
    "TrainingStage",
    "StageConfig",
]
