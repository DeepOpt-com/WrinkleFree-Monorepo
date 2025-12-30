"""Legacy CheaperTraining components.

These modules are no longer actively maintained and have been moved to _legacy.
For new projects, use only the influence and data mixing components from the main module.

Usage:
    from data_handler._legacy.models import MobileLLM, MobileLLMConfig
    from data_handler._legacy.training import Trainer, TrainingStage
"""

from data_handler._legacy.models import (
    MobileLLM,
    MobileLLMConfig,
    MobileLLM140MConfig,
    MobileLLM360MConfig,
    MobileLLM950MConfig,
)
from data_handler._legacy.training.trainer import Trainer
from data_handler._legacy.training.stages import TrainingStage, StageConfig

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
