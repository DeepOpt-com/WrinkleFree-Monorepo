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

Legacy components (moved to cheapertraining._legacy.influence):
- ProbeSetCreator, ProbeDataset
- SelfBoostingFilter, SelfBoostingDataset
"""

# Base interfaces
from cheapertraining.influence.base import (
    EmbeddingExtractor,
    InfluenceCalculator,
    DataSelector,
)

# Config
from cheapertraining.influence.config import (
    InfluenceConfig,
    InfluenceTarget,
    MixtureOptimizationConfig,
    ProbeSetConfig,
    SelfBoostingConfig,
    # New configs for InfluenceDistillation
    JVPEmbeddingConfig,
    LandmarkConfig,
    KRRConfig,
    InfluenceDistillationConfig,
)

# Gradient extraction
from cheapertraining.influence.gradient import (
    DiscriminativeGradientExtractor,
)

# DataInf
from cheapertraining.influence.datainf import (
    DataInfCalculator,
    create_influence_calculator,
)

# Influence Distillation (new)
from cheapertraining.influence.jvp_embedding import (
    JVPEmbeddingExtractor,
)
from cheapertraining.influence.hadamard import (
    RandomizedHadamardTransform,
    create_projection,
)
from cheapertraining.influence.landmark import (
    LandmarkSelector,
    select_landmarks,
)
from cheapertraining.influence.distillation import (
    InfluenceDistillation,
    create_influence_distillation,
)

# Mixture calculation
from cheapertraining.influence.mixture_calculator import (
    MixtureWeightCalculator,
    create_mixture_calculator,
)

# Training integration
from cheapertraining.influence.tracker import (
    InfluenceTracker,
    create_influence_tracker,
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
