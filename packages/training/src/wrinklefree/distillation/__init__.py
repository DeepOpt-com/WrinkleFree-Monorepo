"""Distillation losses for BitDistill training."""

from wrinklefree.distillation.attention_loss import (
    AttentionDistillationLoss,
    AttentionRelationDistillationLoss,
    HiddenStateDistillationLoss,
)
from wrinklefree.distillation.combined_loss import (
    BitDistillLoss,
    ClassificationDistillLoss,
    ContinuePretrainLoss,
)
from wrinklefree.distillation.layerwise_loss import (
    LayerwiseDistillationLoss,
    LayerwiseLossType,
)
from wrinklefree.distillation.logits_loss import (
    LogitsDistillationLoss,
    SoftTargetCrossEntropy,
)
from wrinklefree.distillation.vllm_teacher import (
    VLLMTeacherWrapper,
    VLLMTeacherWithPrefetch,
    VLLMConfig,
    create_vllm_or_inprocess_teacher,
)

__all__ = [
    "LogitsDistillationLoss",
    "SoftTargetCrossEntropy",
    "AttentionDistillationLoss",
    "AttentionRelationDistillationLoss",
    "HiddenStateDistillationLoss",
    "BitDistillLoss",
    "ContinuePretrainLoss",
    "ClassificationDistillLoss",
    # Stage 1.9: Layer-wise distillation
    "LayerwiseDistillationLoss",
    "LayerwiseLossType",
    # vLLM teacher for Stage 3
    "VLLMTeacherWrapper",
    "VLLMTeacherWithPrefetch",
    "VLLMConfig",
    "create_vllm_or_inprocess_teacher",
]
