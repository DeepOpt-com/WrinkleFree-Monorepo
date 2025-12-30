"""Objectives system for unified training.

Provides composable, Hydra-configurable training objectives that can be
combined additively with configurable weights.

Objectives:
- ContinuePretrainObjective: Cross-entropy language modeling (Stage 2)
- LayerwiseDistillationObjective: Hidden state alignment with teacher (Stage 1.9)
- DLMObjective: Block-wise masked language modeling for Fast-dLLM

Manager:
- ObjectiveManager: Combines multiple objectives with weights
- CurriculumScheduler: Adjusts objective weights over training
"""

from wrinklefree.objectives.base import Objective, ObjectiveOutput
from wrinklefree.objectives.continue_pretrain import ContinuePretrainObjective
from wrinklefree.objectives.layerwise import LayerwiseDistillationObjective
from wrinklefree.objectives.manager import ObjectiveManager, CurriculumScheduler
from wrinklefree.objectives.factory import create_objective_manager

__all__ = [
    # Base
    "Objective",
    "ObjectiveOutput",
    # Objectives
    "ContinuePretrainObjective",
    "LayerwiseDistillationObjective",
    # Manager
    "ObjectiveManager",
    "CurriculumScheduler",
    # Factory
    "create_objective_manager",
]
