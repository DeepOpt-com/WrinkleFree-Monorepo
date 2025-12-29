"""Distillation loss functions."""

from distillation.losses.attention_loss import (
    AttentionDistillationLoss,
    AttentionRelationDistillationLoss,
    BitDistillAttentionRelationLoss,
    HiddenStateDistillationLoss,
)
from distillation.losses.combined_loss import (
    BitDistillLoss,
    ClassificationDistillLoss,
)
from distillation.losses.logits_loss import (
    LogitsDistillationLoss,
    SoftTargetCrossEntropy,
)

__all__ = [
    # Logits
    "LogitsDistillationLoss",
    "SoftTargetCrossEntropy",
    # Attention
    "AttentionDistillationLoss",
    "AttentionRelationDistillationLoss",
    "BitDistillAttentionRelationLoss",
    "HiddenStateDistillationLoss",
    # Combined
    "BitDistillLoss",
    "ClassificationDistillLoss",
]
