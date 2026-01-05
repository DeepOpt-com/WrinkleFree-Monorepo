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

# Re-export from _legacy for backwards compatibility (direct import to avoid _legacy/__init__.py)
from data_handler._legacy.influence_datainf import (
    # Base interfaces
    EmbeddingExtractor,
    InfluenceCalculator,
    DataSelector,
    # Configs
    InfluenceConfig,
    InfluenceTarget,
    MixtureOptimizationConfig,
    ProbeSetConfig,
    SelfBoostingConfig,
    JVPEmbeddingConfig,
    LandmarkConfig,
    KRRConfig,
    InfluenceDistillationConfig,
    # Gradient extraction
    DiscriminativeGradientExtractor,
    # DataInf
    DataInfCalculator,
    create_influence_calculator,
    # Influence Distillation
    JVPEmbeddingExtractor,
    RandomizedHadamardTransform,
    create_projection,
    LandmarkSelector,
    select_landmarks,
    InfluenceDistillation,
    create_influence_distillation,
    # Mixture calculation
    MixtureWeightCalculator,
    create_mixture_calculator,
    # Tracker
    InfluenceTracker,
    create_influence_tracker,
)

__all__ = [
    "EmbeddingExtractor",
    "InfluenceCalculator",
    "DataSelector",
    "InfluenceConfig",
    "InfluenceTarget",
    "MixtureOptimizationConfig",
    "ProbeSetConfig",
    "SelfBoostingConfig",
    "JVPEmbeddingConfig",
    "LandmarkConfig",
    "KRRConfig",
    "InfluenceDistillationConfig",
    "DiscriminativeGradientExtractor",
    "DataInfCalculator",
    "create_influence_calculator",
    "JVPEmbeddingExtractor",
    "RandomizedHadamardTransform",
    "create_projection",
    "LandmarkSelector",
    "select_landmarks",
    "InfluenceDistillation",
    "create_influence_distillation",
    "MixtureWeightCalculator",
    "create_mixture_calculator",
    "InfluenceTracker",
    "create_influence_tracker",
]
