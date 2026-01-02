"""DEPRECATED: Distillation losses for legacy training.

This module is deprecated and kept only for backward compatibility with:
- Stage 1.9 layer-wise distillation (objectives/layerwise.py uses LayerwiseDistillationLoss)
- Legacy training code

For new code, use the objectives system directly:
    from wrinklefree.objectives import LayerwiseDistillationObjective, ContinuePretrainObjective

Stage 3 distillation has been moved to the separate `distillation` package.
"""

import warnings

warnings.warn(
    "wrinklefree.distillation is deprecated. "
    "Use wrinklefree.objectives for new code. "
    "This module is kept only for backward compatibility.",
    DeprecationWarning,
    stacklevel=2,
)

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
