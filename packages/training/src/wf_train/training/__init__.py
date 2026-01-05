"""Training utilities for BitNet models.

Active utilities:
- FSDP wrappers and distributed training helpers
- Auto-setup for checkpoint resolution and BitNet conversion
- MuonClip patches for FSDP compatibility

For training, use:
    from wf_train.lightning import WrinkleFreeLightningModule, WrinkleFreeDataModule
"""

# Active utilities (still used by Lightning and other code)
from wf_train.training.fsdp_wrapper import (
    apply_activation_checkpointing,
    cleanup_distributed,
    get_fsdp_state_dict,
    get_mixed_precision_policy,
    get_sharding_strategy,
    load_fsdp_state_dict,
    setup_distributed,
    wrap_model_fsdp,
)

__all__ = [
    "wrap_model_fsdp",
    "apply_activation_checkpointing",
    "get_sharding_strategy",
    "get_mixed_precision_policy",
    "get_fsdp_state_dict",
    "load_fsdp_state_dict",
    "setup_distributed",
    "cleanup_distributed",
]
