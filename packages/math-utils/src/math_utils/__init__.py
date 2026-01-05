"""math-utils: Pure mathematical utilities for influence functions and gradient computation."""

from math_utils.influence import (
    # Base interfaces
    EmbeddingExtractor as EmbeddingExtractor,
    InfluenceCalculator as InfluenceCalculator,
    DataSelector as DataSelector,
    # Configs
    InfluenceConfig as InfluenceConfig,
    InfluenceTarget as InfluenceTarget,
    JVPEmbeddingConfig as JVPEmbeddingConfig,
    LandmarkConfig as LandmarkConfig,
    KRRConfig as KRRConfig,
    InfluenceDistillationConfig as InfluenceDistillationConfig,
    MixtureOptimizationConfig as MixtureOptimizationConfig,
    ProbeSetConfig as ProbeSetConfig,
    SelfBoostingConfig as SelfBoostingConfig,
    # Core algorithms
    DataInfCalculator as DataInfCalculator,
    create_influence_calculator as create_influence_calculator,
    DiscriminativeGradientExtractor as DiscriminativeGradientExtractor,
    JVPEmbeddingExtractor as JVPEmbeddingExtractor,
    RandomizedHadamardTransform as RandomizedHadamardTransform,
    create_projection as create_projection,
    LandmarkSelector as LandmarkSelector,
    select_landmarks as select_landmarks,
    InfluenceDistillation as InfluenceDistillation,
    create_influence_distillation as create_influence_distillation,
)

__all__ = [
    # Base
    "EmbeddingExtractor",
    "InfluenceCalculator",
    "DataSelector",
    # Configs
    "InfluenceConfig",
    "InfluenceTarget",
    "JVPEmbeddingConfig",
    "LandmarkConfig",
    "KRRConfig",
    "InfluenceDistillationConfig",
    "MixtureOptimizationConfig",
    "ProbeSetConfig",
    "SelfBoostingConfig",
    # Core
    "DataInfCalculator",
    "create_influence_calculator",
    "DiscriminativeGradientExtractor",
    "JVPEmbeddingExtractor",
    "RandomizedHadamardTransform",
    "create_projection",
    "LandmarkSelector",
    "select_landmarks",
    "InfluenceDistillation",
    "create_influence_distillation",
]
