"""Objectives system for unified training.

Provides composable, Hydra-configurable training objectives that can be
combined additively with configurable weights.

Objectives:
- ContinuePretrainObjective: Cross-entropy language modeling
- SFTObjective: Supervised fine-tuning (instruction-masked CE loss)
- DistillObjective: Unified distillation (hidden, logits, attention, LRC)

Manager:
- ObjectiveManager: Combines multiple objectives with weights
- CurriculumScheduler: Adjusts objective weights over training
"""

from wf_train.objectives.base import Objective, ObjectiveOutput
from wf_train.objectives.continue_pretrain import ContinuePretrainObjective
from wf_train.objectives.sft import SFTObjective
from wf_train.objectives.distill import (
    DistillObjective,
    LayerWiseConfig,
    HiddenConfig,  # Deprecated alias for LayerWiseConfig
    LRCConfig,  # Deprecated alias for LayerWiseConfig
    LogitsConfig,
    AttentionConfig,
    HiddenLossType,
    LogitsMode,
    AttentionMode,
)
from wf_train.objectives.manager import ObjectiveManager, CurriculumScheduler
from wf_train.objectives.factory import create_objective_manager

__all__ = [
    # Base
    "Objective",
    "ObjectiveOutput",
    # Objectives
    "ContinuePretrainObjective",
    "SFTObjective",
    "DistillObjective",
    # Distill configs
    "LayerWiseConfig",  # Unified config for hidden/lrc
    "HiddenConfig",  # Deprecated alias
    "LRCConfig",  # Deprecated alias
    "LogitsConfig",
    "AttentionConfig",
    # Distill enums
    "HiddenLossType",
    "LogitsMode",
    "AttentionMode",
    # Manager
    "ObjectiveManager",
    "CurriculumScheduler",
    # Factory
    "create_objective_manager",
]
