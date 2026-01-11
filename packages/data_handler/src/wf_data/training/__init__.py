"""Training infrastructure - active components.

Legacy components (Trainer, TrainingStage, schedulers) have been moved to
cheapertraining._legacy.training
"""

from wf_data.training.early_stopping import PlateauEarlyStopping
from wf_data.training.gradient_clipping import (
    ZClip,
    ZClipStats,
    clip_grad_with_zclip,
)
from wf_data.training.optimizer import (
    InfluenceAwareOptimizer,
    create_optimizer,
    get_parameter_groups,
    get_num_parameters,
)
from wf_data.training.qk_clip import (
    apply_qk_clip,
    QKClipStats,
    QKClipMonitor,
)

__all__ = [
    # Gradient clipping
    "ZClip",
    "ZClipStats",
    "clip_grad_with_zclip",
    # Early stopping
    "PlateauEarlyStopping",
    # Optimizer utilities
    "InfluenceAwareOptimizer",
    "create_optimizer",
    "get_parameter_groups",
    "get_num_parameters",
    # QK clipping
    "apply_qk_clip",
    "QKClipStats",
    "QKClipMonitor",
]
