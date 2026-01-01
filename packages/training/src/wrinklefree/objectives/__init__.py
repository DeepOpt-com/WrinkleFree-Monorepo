"""Objectives system for unified training.

Provides composable, Hydra-configurable training objectives that can be
combined additively with configurable weights.

Objectives:
- ContinuePretrainObjective: Cross-entropy language modeling (Stage 2)
- LayerwiseDistillationObjective: Hidden state alignment with teacher (Stage 1.9)
- DLMObjective: Block-wise masked language modeling for Fast-dLLM
- LogitsDistillationObjective: KL divergence on teacher/student logits (BitDistill)
- AttentionRelationDistillationObjective: Attention relation distillation (BitDistill)
- TCSDistillationObjective: Target Concrete Score for DLM students
- BlockAttentionDistillationObjective: Block-wise attention for AR->DLM
- BitDistillObjective: Combined BitDistill (logits + attention)
- LRCReconstructionObjective: Low-Rank Correction for quantized models

Manager:
- ObjectiveManager: Combines multiple objectives with weights
- CurriculumScheduler: Adjusts objective weights over training
"""

from wrinklefree.objectives.base import Objective, ObjectiveOutput
from wrinklefree.objectives.continue_pretrain import ContinuePretrainObjective
from wrinklefree.objectives.dlm import DLMObjective
from wrinklefree.objectives.layerwise import LayerwiseDistillationObjective
from wrinklefree.objectives.logits_distill import LogitsDistillationObjective
from wrinklefree.objectives.attention_distill import AttentionRelationDistillationObjective
from wrinklefree.objectives.tcs_distill import TCSDistillationObjective
from wrinklefree.objectives.block_attention_distill import BlockAttentionDistillationObjective
from wrinklefree.objectives.bitdistill import BitDistillObjective
from wrinklefree.objectives.lrc_reconstruction import LRCReconstructionObjective
from wrinklefree.objectives.manager import ObjectiveManager, CurriculumScheduler
from wrinklefree.objectives.factory import create_objective_manager

__all__ = [
    # Base
    "Objective",
    "ObjectiveOutput",
    # Objectives
    "ContinuePretrainObjective",
    "DLMObjective",
    "LayerwiseDistillationObjective",
    # Distillation objectives
    "LogitsDistillationObjective",
    "AttentionRelationDistillationObjective",
    "TCSDistillationObjective",
    "BlockAttentionDistillationObjective",
    "BitDistillObjective",
    # LRC (Low-Rank Correction)
    "LRCReconstructionObjective",
    # Manager
    "ObjectiveManager",
    "CurriculumScheduler",
    # Factory
    "create_objective_manager",
]
