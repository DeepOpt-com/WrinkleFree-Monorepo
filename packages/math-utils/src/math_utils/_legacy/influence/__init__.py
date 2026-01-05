"""LEGACY: Influence function algorithms for data valuation and selection.

DEPRECATED: Use training.meta_optimization.odm instead (O(1) complexity).
See https://arxiv.org/abs/2312.02406 for the ODM paper.
"""

# Use relative imports within _legacy
from .base import (
    EmbeddingExtractor as EmbeddingExtractor,
    InfluenceCalculator as InfluenceCalculator,
    DataSelector as DataSelector,
)
from .config import (
    InfluenceConfig as InfluenceConfig,
    InfluenceTarget as InfluenceTarget,
    JVPEmbeddingConfig as JVPEmbeddingConfig,
    LandmarkConfig as LandmarkConfig,
    KRRConfig as KRRConfig,
    InfluenceDistillationConfig as InfluenceDistillationConfig,
    MixtureOptimizationConfig as MixtureOptimizationConfig,
    ProbeSetConfig as ProbeSetConfig,
    SelfBoostingConfig as SelfBoostingConfig,
)
from .gradient import (
    DiscriminativeGradientExtractor as DiscriminativeGradientExtractor,
)
from .datainf import (
    DataInfCalculator as DataInfCalculator,
    create_influence_calculator as create_influence_calculator,
)
from .jvp_embedding import (
    JVPEmbeddingExtractor as JVPEmbeddingExtractor,
)
from .hadamard import (
    RandomizedHadamardTransform as RandomizedHadamardTransform,
    create_projection as create_projection,
)
from .landmark import (
    LandmarkSelector as LandmarkSelector,
    select_landmarks as select_landmarks,
)
from .distillation import (
    InfluenceDistillation as InfluenceDistillation,
    create_influence_distillation as create_influence_distillation,
)
from .meta_gradient import (
    MetaGradientCalculator as MetaGradientCalculator,
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
    # Meta-gradient
    "MetaGradientCalculator",
]
