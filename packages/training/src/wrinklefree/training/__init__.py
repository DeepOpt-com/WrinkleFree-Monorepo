"""Training utilities for BitNet models.

Active utilities:
- FSDP wrappers and distributed training helpers
- Auto-setup for checkpoint resolution and BitNet conversion
- MuonClip patches for FSDP compatibility

DEPRECATED (moved to _legacy/):
- Trainer, ContinuedPretrainingTrainer: Use PyTorch Lightning instead
- stage1.py: Use bitnet_arch.auto_convert_if_needed() instead

For new training code, use:
    from wrinklefree.lightning import WrinkleFreeLightningModule, WrinkleFreeDataModule
"""

# Active utilities (still used by Lightning and other code)
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

# Re-export legacy classes for backward compatibility (with deprecation warning)
from wrinklefree.training._legacy import (
    Trainer,
    create_optimizer,
    create_scheduler,
    download_checkpoint_from_gcs,
    ContinuedPretrainingTrainer,
    Stage2Trainer,
    run_stage2,
    convert_model_to_bitnet,
    run_stage1,
)

__all__ = [
    # FSDP utilities (ACTIVE)
    "wrap_model_fsdp",
    "apply_activation_checkpointing",
    "get_sharding_strategy",
    "get_mixed_precision_policy",
    "get_fsdp_state_dict",
    "load_fsdp_state_dict",
    "setup_distributed",
    "cleanup_distributed",
    # DEPRECATED - kept for backward compatibility
    # Use PyTorch Lightning instead
    "Trainer",
    "create_optimizer",
    "create_scheduler",
    "download_checkpoint_from_gcs",
    "ContinuedPretrainingTrainer",
    "Stage2Trainer",
    "run_stage2",
    # DEPRECATED - use bitnet_arch.auto_convert_if_needed()
    "convert_model_to_bitnet",
    "run_stage1",
]
