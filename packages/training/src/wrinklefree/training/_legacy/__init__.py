"""LEGACY: Training utilities that have been superseded by PyTorch Lightning.

These modules are kept for backward compatibility with existing tests and benchmarks.
For new code, use the Lightning-based training pipeline:

    from wrinklefree.lightning import WrinkleFreeLightningModule, WrinkleFreeDataModule

DEPRECATED modules in this package:
- trainer.py: Base Trainer class (use PyTorch Lightning Trainer instead)
- continued_pretraining.py: ContinuedPretrainingTrainer (use WrinkleFreeLightningModule)
- stage1.py: Stage 1 SubLN insertion (use bitnet_arch.auto_convert_if_needed)
"""

import warnings

warnings.warn(
    "wrinklefree.training._legacy is deprecated. "
    "Use wrinklefree.lightning for new training code. "
    "These modules are kept only for backward compatibility with tests.",
    DeprecationWarning,
    stacklevel=2,
)

from wrinklefree.training._legacy.trainer import (
    Trainer,
    create_optimizer,
    create_scheduler,
    download_checkpoint_from_gcs,
)
from wrinklefree.training._legacy.continued_pretraining import (
    ContinuedPretrainingTrainer,
    run_stage2,
)
from wrinklefree.training._legacy.stage1 import (
    convert_model_to_bitnet,
    run_stage1,
)

# Backward compatibility alias
Stage2Trainer = ContinuedPretrainingTrainer

__all__ = [
    # Base Trainer (DEPRECATED - use Lightning)
    "Trainer",
    "create_optimizer",
    "create_scheduler",
    "download_checkpoint_from_gcs",
    # Continued Pre-training (DEPRECATED - use Lightning)
    "ContinuedPretrainingTrainer",
    "Stage2Trainer",
    "run_stage2",
    # Stage 1: BitNet conversion (DEPRECATED - use bitnet_arch.auto_convert_if_needed)
    "convert_model_to_bitnet",
    "run_stage1",
]
