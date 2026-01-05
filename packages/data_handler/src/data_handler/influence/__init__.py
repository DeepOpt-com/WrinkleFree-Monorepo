"""DEPRECATED: DataInf-based influence tracking.

This module has been replaced by ODM (Online Data Mixing) in the meta-optimization system.
ODM uses EXP3 multi-armed bandit with O(1) complexity instead of O(K) gradient computation.

Use `training.meta_optimization.odm` instead of `training.influence`.

Reference: https://arxiv.org/abs/2312.02406 (ODM)

For legacy code that still needs DataInf, imports are redirected to _legacy.
"""

import warnings

warnings.warn(
    "data_handler.influence is deprecated. Use training.meta_optimization.odm instead. "
    "See https://arxiv.org/abs/2312.02406 for the ODM paper.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from _legacy for backwards compatibility
from data_handler._legacy.influence_datainf import *  # noqa: F401, F403
