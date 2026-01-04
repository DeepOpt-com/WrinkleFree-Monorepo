"""Influence functions module - active components for data-efficient training.

Implements influence function methodology from:
- DataInf (ICLR 2024) - Efficient influence without Hessian inversion
- MobileLLM-R1 (arXiv:2509.24945) - Cross-domain influence methodology
- InfluenceDistillation (arXiv:2505.19051) - Landmark-based influence approximation

Key components:
- InfluenceTracker: Training callback for dynamic weight updates (primary API)
- DataInfCalculator: Tractable influence calculation without Hessian inversion
- InfluenceDistillation: Landmark-based influence approximation (faster for large N)
- MixtureWeightCalculator: Optimize pre-training dataset mixture weights
- DiscriminativeGradientExtractor: Extract gradients from discriminative layers
- JVPEmbeddingExtractor: Extract JVP embeddings from transformer layers

Note: Pure math components are now in math_utils package.
This module re-exports them for backward compatibility.
"""

# Re-export from math_utils for backward compatibility (using explicit 'as X' for type checkers)
from math_utils.influence import (
    # Base interfaces
    EmbeddingExtractor as EmbeddingExtractor,
    InfluenceCalculator as InfluenceCalculator,
    DataSelector as DataSelector,
    # Configs
    InfluenceConfig as InfluenceConfig,
    InfluenceTarget as InfluenceTarget,
    MixtureOptimizationConfig as MixtureOptimizationConfig,
    ProbeSetConfig as ProbeSetConfig,
    SelfBoostingConfig as SelfBoostingConfig,
    JVPEmbeddingConfig as JVPEmbeddingConfig,
    LandmarkConfig as LandmarkConfig,
    KRRConfig as KRRConfig,
    InfluenceDistillationConfig as InfluenceDistillationConfig,
    # Gradient extraction
    DiscriminativeGradientExtractor as DiscriminativeGradientExtractor,
    # DataInf
    DataInfCalculator as DataInfCalculator,
    create_influence_calculator as create_influence_calculator,
    # Influence Distillation
    JVPEmbeddingExtractor as JVPEmbeddingExtractor,
    RandomizedHadamardTransform as RandomizedHadamardTransform,
    create_projection as create_projection,
    LandmarkSelector as LandmarkSelector,
    select_landmarks as select_landmarks,
    InfluenceDistillation as InfluenceDistillation,
    create_influence_distillation as create_influence_distillation,
)

# Integration components (stay in data_handler)
from data_handler.influence.mixture_calculator import (
    MixtureWeightCalculator as MixtureWeightCalculator,
    create_mixture_calculator as create_mixture_calculator,
)
from data_handler.influence.tracker import (
    InfluenceTracker as InfluenceTracker,
    create_influence_tracker as create_influence_tracker,
)

__all__ = [
    # Base interfaces
    "EmbeddingExtractor",
    "InfluenceCalculator",
    "DataSelector",
    # Primary API (use this for training integration)
    "InfluenceTracker",
    "create_influence_tracker",
    # Config - DataInf
    "InfluenceConfig",
    "InfluenceTarget",
    "MixtureOptimizationConfig",
    "ProbeSetConfig",
    "SelfBoostingConfig",
    # Config - InfluenceDistillation
    "JVPEmbeddingConfig",
    "LandmarkConfig",
    "KRRConfig",
    "InfluenceDistillationConfig",
    # Gradient extraction
    "DiscriminativeGradientExtractor",
    # DataInf
    "DataInfCalculator",
    "create_influence_calculator",
    # Influence Distillation
    "JVPEmbeddingExtractor",
    "RandomizedHadamardTransform",
    "create_projection",
    "LandmarkSelector",
    "select_landmarks",
    "InfluenceDistillation",
    "create_influence_distillation",
    # Mixture calculation
    "MixtureWeightCalculator",
    "create_mixture_calculator",
]
