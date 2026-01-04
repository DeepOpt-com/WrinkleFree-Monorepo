"""Influence function algorithms for data valuation and selection."""

from math_utils.influence.base import (
    EmbeddingExtractor as EmbeddingExtractor,
    InfluenceCalculator as InfluenceCalculator,
    DataSelector as DataSelector,
)
from math_utils.influence.config import (
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
from math_utils.influence.gradient import (
    DiscriminativeGradientExtractor as DiscriminativeGradientExtractor,
)
from math_utils.influence.datainf import (
    DataInfCalculator as DataInfCalculator,
    create_influence_calculator as create_influence_calculator,
)
from math_utils.influence.jvp_embedding import (
    JVPEmbeddingExtractor as JVPEmbeddingExtractor,
)
from math_utils.influence.hadamard import (
    RandomizedHadamardTransform as RandomizedHadamardTransform,
    create_projection as create_projection,
)
from math_utils.influence.landmark import (
    LandmarkSelector as LandmarkSelector,
    select_landmarks as select_landmarks,
)
from math_utils.influence.distillation import (
    InfluenceDistillation as InfluenceDistillation,
    create_influence_distillation as create_influence_distillation,
)
from math_utils.influence.meta_gradient import (
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
