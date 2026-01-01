"""Distillation losses for BitDistill training.

NOTE: Stage 3 distillation has been moved to the separate `distillation` package.
This module now only contains:
- LayerwiseDistillationLoss - Used by Stage 1.9 layer-wise distillation
- ContinuePretrainLoss - Used by Stage 2 continue pre-training

For Stage 3 distillation (logits + attention), use:
    from distillation import BitDistillLoss, LocalTeacher, DistillationTrainer
"""

from wrinklefree.distillation.combined_loss import ContinuePretrainLoss
from wrinklefree.distillation.layerwise_loss import (
    LayerwiseDistillationLoss,
    LayerwiseLossType,
)

__all__ = [
    # Stage 1.9: Layer-wise distillation
    "LayerwiseDistillationLoss",
    "LayerwiseLossType",
    # Stage 2: Continue pre-training
    "ContinuePretrainLoss",
]
