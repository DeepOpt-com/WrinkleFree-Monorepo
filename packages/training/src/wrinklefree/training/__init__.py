"""Training utilities for BitNet models."""

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
# HiddenStateTeacherWrapper moved to distillation module
from wrinklefree.distillation import HiddenStateTeacherWrapper
# Stage19Trainer merged into Stage2Trainer
from wrinklefree.training.stage2 import Stage2Trainer, run_stage2
# Backward compatibility: run_stage1_9 is deprecated, use run_stage2 with pre_stage_2.enabled=true
from wrinklefree.training.stage1_9 import run_stage1_9  # Deprecated wrapper
from wrinklefree.training.stage3 import Stage3Trainer, TeacherWrapper, run_stage3
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
    # Stage 1
    "convert_model_to_bitnet",
    "run_stage1",
    # Stage 1.9: Layer-wise distillation (deprecated - use run_stage2 with pre_stage_2.enabled=true)
    "HiddenStateTeacherWrapper",
    "run_stage1_9",  # Deprecated wrapper for backward compatibility
    # Stage 2
    "Stage2Trainer",
    "run_stage2",
    # Stage 3
    "TeacherWrapper",
    "Stage3Trainer",
    "run_stage3",
]
