"""CheaperTraining: Influence-based data selection for LLM training.

Active components:
- DataInfCalculator: Tractable influence calculation without Hessian inversion
- MixtureWeightCalculator: Optimize dataset mixture weights
- InfluenceAwareOptimizer: Optimizer with dynamic mixture updates
- DiscriminativeGradientExtractor: Gradient extraction from discriminative layers
- InfluenceConfig: Configuration for influence calculation
- MixedDataset: Dataset with dynamic mixture weights

Legacy components (moved to cheapertraining._legacy):
- MobileLLM, MobileLLMConfig: Use from data_handler._legacy.models
- Trainer, TrainingStage: Use from data_handler._legacy.training
- SelfBoostingFilter: Use from data_handler._legacy.influence
"""

__version__ = "0.1.0"

# Active components only - used by WrinkleFree-1.58Quant
from data_handler.influence.config import InfluenceConfig
from data_handler.influence.gradient import DiscriminativeGradientExtractor
from data_handler.influence.datainf import DataInfCalculator
from data_handler.influence.mixture_calculator import MixtureWeightCalculator
from data_handler.data.mixing import MixedDataset
from data_handler.training.optimizer import InfluenceAwareOptimizer

__all__ = [
    "InfluenceConfig",
    "DiscriminativeGradientExtractor",
    "DataInfCalculator",
    "MixtureWeightCalculator",
    "MixedDataset",
    "InfluenceAwareOptimizer",
]
