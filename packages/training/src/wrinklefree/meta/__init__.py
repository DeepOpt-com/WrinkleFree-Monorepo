"""Meta-optimization outer loop for Lightning trainer.

This module provides configurable meta-optimization that jointly optimizes:
- Dataset mixture weights
- Objective weights (CE, DLM, distillation, etc.)
- Learning rate scales (per parameter group)

Using influence-based gradient estimation with multi-objective Pareto optimization.

References:
- LibMOON (NeurIPS 2024): https://arxiv.org/abs/2409.02969
  Gradient-based multi-objective optimization with MGDA/EPO
- ScaleBiO (2024): https://arxiv.org/abs/2406.19976
  Scalable bi-level optimization for LLM data reweighting
- DataInf (ICLR 2024): https://openreview.net/forum?id=9m02ib92Wz
  Tractable influence without Hessian inversion
"""

from wrinklefree.meta.config import (
    MetaConstraintsConfig,
    MetaGradientConfig,
    MetaOptimizationConfig,
    ParetoConfig,
    ValidationObjectiveConfig,
)
from wrinklefree.meta.manager import MetaParameterManager
from wrinklefree.meta.pareto import ParetoGradientSolver
from wrinklefree.meta.callback import MetaOptimizerCallback

__all__ = [
    # Config classes
    "MetaOptimizationConfig",
    "ParetoConfig",
    "ValidationObjectiveConfig",
    "MetaConstraintsConfig",
    "MetaGradientConfig",
    # Core classes
    "MetaParameterManager",
    "ParetoGradientSolver",
    "MetaOptimizerCallback",
]
