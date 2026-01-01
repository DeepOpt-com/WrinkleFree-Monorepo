"""Training utilities for BitNet models.

Unified training with composable objectives:
- ContinuedPretrainingTrainer: Main trainer class (replaces Stage2Trainer)
- ObjectiveManager: Combines multiple objectives with configurable weights
- On-the-fly BitNet conversion via bitnet_arch

NOTE: Stage 3 distillation has been moved to the separate `distillation` package.
For Stage 3, use:
    uv run --package wrinklefree-distillation python scripts/distill.py \
        student.checkpoint_path=outputs/stage2/checkpoint.pt
"""

from wrinklefree.training.fsdp_wrapper import (
    apply_activation_checkpointing,
    cleanup_distributed,
    get_fsdp_state_dict,
    get_mixed_precision_policy,
    get_sharding_strategy,
    load_fsdp_state_dict,
    setup_distributed,
    wrap_model_fsdp,
)
from wrinklefree.training.stage1 import convert_model_to_bitnet, run_stage1
# ContinuedPretrainingTrainer is the new name for Stage2Trainer
from wrinklefree.training.continued_pretraining import ContinuedPretrainingTrainer, run_stage2
# Backward compatibility alias
Stage2Trainer = ContinuedPretrainingTrainer
# Backward compatibility: run_stage1_9 is deprecated, use run_stage2 with pre_stage_2.enabled=true
from wrinklefree.training.stage1_9 import run_stage1_9  # Deprecated wrapper
from wrinklefree.training.trainer import (
    Trainer,
    create_optimizer,
    create_scheduler,
    download_checkpoint_from_gcs,
)

__all__ = [
    # FSDP utilities
    "wrap_model_fsdp",
    "apply_activation_checkpointing",
    "get_sharding_strategy",
    "get_mixed_precision_policy",
    "get_fsdp_state_dict",
    "load_fsdp_state_dict",
    "setup_distributed",
    "cleanup_distributed",
    # Trainer
    "Trainer",
    "create_optimizer",
    "create_scheduler",
    "download_checkpoint_from_gcs",
    # Stage 1: BitNet conversion (legacy - prefer bitnet_arch.auto_convert_if_needed)
    "convert_model_to_bitnet",
    "run_stage1",
    # Stage 1.9: Layer-wise distillation (deprecated - use run_stage2 with pre_stage_2.enabled=true)
    "run_stage1_9",  # Deprecated wrapper for backward compatibility
    # Continued Pre-training (unified trainer)
    "ContinuedPretrainingTrainer",
    "Stage2Trainer",  # Backward compatibility alias
    "run_stage2",
    # Stage 3: Moved to distillation package
    # Use: uv run --package wrinklefree-distillation python scripts/distill.py
]
