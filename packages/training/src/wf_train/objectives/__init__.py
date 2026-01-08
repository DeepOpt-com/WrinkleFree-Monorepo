"""Objectives system for unified training.

Provides composable, Hydra-configurable training objectives that can be
combined additively with configurable weights.

Objectives:
- ContinuePretrainObjective: Cross-entropy language modeling
- DLMObjective: Block-wise masked language modeling for Fast-dLLM
- SFTObjective: Supervised fine-tuning (instruction-masked CE loss)
- DistillObjective: Unified distillation (hidden, logits, attention, LRC)

Manager:
- ObjectiveManager: Combines multiple objectives with weights
- CurriculumScheduler: Adjusts objective weights over training
"""

from wf_train.objectives.base import Objective, ObjectiveOutput
from wf_train.objectives.continue_pretrain import ContinuePretrainObjective
from wf_train.objectives.dlm import DLMObjective
from wf_train.objectives.sft import SFTObjective
from wf_train.objectives.distill import (
    DistillObjective,
    HiddenConfig,
    LogitsConfig,
    AttentionConfig,
    LRCConfig,
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
    "DLMObjective",
    "SFTObjective",
    "DistillObjective",
    # Distill configs
    "HiddenConfig",
    "LogitsConfig",
    "AttentionConfig",
    "LRCConfig",
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
