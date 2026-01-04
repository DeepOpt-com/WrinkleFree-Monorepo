"""Efficient meta-optimization for Lightning trainer.

This module provides O(1) complexity meta-optimization using:
- LDC-MTL: Objective weight optimization (CE vs DLM vs distillation)
- ODM/EXP3: Dataset weight optimization (bandit-based data mixing)

Both methods are efficient, principled, and require no external dependencies.

References:
- LDC-MTL (2025): https://arxiv.org/abs/2502.08585
  Loss Discrepancy Control for Multi-Task Learning
- ODM (2023): https://arxiv.org/abs/2312.02406
  Efficient Online Data Mixing For Language Model Pre-Training
"""

from wrinklefree.meta.config import (
    LDCMTLConfig,
    MetaOptimizationConfig,
    ODMConfig,
)
from wrinklefree.meta.ldc_mtl import (
    LDCMTLManager,
    ObjectiveRouter,
    compute_loss_discrepancy,
)
from wrinklefree.meta.odm import OnlineDataMixer
from wrinklefree.meta.callback import MetaOptimizerCallback

__all__ = [
    # Config classes
    "MetaOptimizationConfig",
    "LDCMTLConfig",
    "ODMConfig",
    # LDC-MTL components
    "LDCMTLManager",
    "ObjectiveRouter",
    "compute_loss_discrepancy",
    # ODM components
    "OnlineDataMixer",
    # Callback
    "MetaOptimizerCallback",
]
