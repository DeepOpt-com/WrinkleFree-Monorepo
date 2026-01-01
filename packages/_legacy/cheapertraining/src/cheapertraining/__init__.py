"""CheaperTraining: Influence-based data selection for LLM training.

Active components:
- DataInfCalculator: Tractable influence calculation without Hessian inversion
- MixtureWeightCalculator: Optimize dataset mixture weights
- InfluenceAwareOptimizer: Optimizer with dynamic mixture updates
- DiscriminativeGradientExtractor: Gradient extraction from discriminative layers
- InfluenceConfig: Configuration for influence calculation
- MixedDataset: Dataset with dynamic mixture weights

Legacy components (moved to cheapertraining._legacy):
- MobileLLM, MobileLLMConfig: Use from cheapertraining._legacy.models
- Trainer, TrainingStage: Use from cheapertraining._legacy.training
- SelfBoostingFilter: Use from cheapertraining._legacy.influence
"""

__version__ = "0.1.0"

# Active components only - used by WrinkleFree-1.58Quant
from cheapertraining.influence.config import InfluenceConfig
from cheapertraining.influence.gradient import DiscriminativeGradientExtractor
from cheapertraining.influence.datainf import DataInfCalculator
from cheapertraining.influence.mixture_calculator import MixtureWeightCalculator
from cheapertraining.data.mixing import MixedDataset
from cheapertraining.training.optimizer import InfluenceAwareOptimizer

__all__ = [
    "InfluenceConfig",
    "DiscriminativeGradientExtractor",
    "DataInfCalculator",
    "MixtureWeightCalculator",
    "MixedDataset",
    "InfluenceAwareOptimizer",
]
