"""FSDP2 distributed training utilities.

Reference: PyTorch TorchTitan and FSDP2 with DTensor
"""

from typing import Optional, Dict, Any
from functools import partial

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh


def apply_fsdp2(
    model: nn.Module,
    device_mesh: Optional[DeviceMesh] = None,
    mixed_precision: bool = True,
    use_orig_params: bool = True,
    activation_checkpointing: str = "none",
    cpu_offload: bool = False,
) -> nn.Module:
    """Apply FSDP2 to model.

    Args:
        model: Model to wrap with FSDP2
        device_mesh: Device mesh for sharding (created if None)
        mixed_precision: Whether to use mixed precision
        use_orig_params: Use original parameter references
        activation_checkpointing: Checkpointing strategy ('none', 'selective', 'full')
        cpu_offload: Whether to offload to CPU

    Returns:
        FSDP2-wrapped model
    """
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    except ImportError:
        raise ImportError("FSDP requires PyTorch 2.0+")

    # Create device mesh if not provided
    if device_mesh is None:
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            device_mesh = DeviceMesh("cuda", torch.arange(world_size))
        else:
            # Single GPU fallback
            return model

    # Mixed precision policy
    mp_policy = None
    if mixed_precision:
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
        )

    # Auto wrap policy for transformer layers
    from cheapertraining._legacy.models.transformer import TransformerBlock

    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )

    # Apply activation checkpointing before FSDP wrapping
    if activation_checkpointing == "selective":
        _apply_selective_checkpointing(model)
    elif activation_checkpointing == "full":
        _apply_full_checkpointing(model)

    # Wrap with FSDP
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp_policy,
        auto_wrap_policy=auto_wrap_policy,
        use_orig_params=use_orig_params,
        device_id=torch.cuda.current_device(),
    )

    return model


def _apply_selective_checkpointing(model: nn.Module):
    """Apply activation checkpointing to attention and FFN layers."""
    from torch.utils.checkpoint import checkpoint_sequential

    for module in model.modules():
        if hasattr(module, 'attention'):
            module.attention = _make_checkpointed(module.attention)
        if hasattr(module, 'ffn'):
            module.ffn = _make_checkpointed(module.ffn)


def _apply_full_checkpointing(model: nn.Module):
    """Apply activation checkpointing to all transformer blocks."""
    from cheapertraining._legacy.models.transformer import TransformerBlock

    for module in model.modules():
        if isinstance(module, TransformerBlock):
            module.forward = _make_checkpointed_forward(module.forward)


def _make_checkpointed(module: nn.Module) -> nn.Module:
    """Wrap module with checkpointing."""
    from torch.utils.checkpoint import checkpoint

    original_forward = module.forward

    def checkpointed_forward(*args, **kwargs):
        return checkpoint(original_forward, *args, use_reentrant=False, **kwargs)

    module.forward = checkpointed_forward
    return module


def _make_checkpointed_forward(forward_fn):
    """Create checkpointed forward function."""
    from torch.utils.checkpoint import checkpoint

    def checkpointed_forward(*args, **kwargs):
        return checkpoint(forward_fn, *args, use_reentrant=False, **kwargs)

    return checkpointed_forward


class FSDP2Config:
    """Configuration for FSDP2."""

    def __init__(
        self,
        sharding_strategy: str = "FULL_SHARD",
        mixed_precision: bool = True,
        use_orig_params: bool = True,
        activation_checkpointing: str = "selective",
        cpu_offload: bool = False,
    ):
        self.sharding_strategy = sharding_strategy
        self.mixed_precision = mixed_precision
        self.use_orig_params = use_orig_params
        self.activation_checkpointing = activation_checkpointing
        self.cpu_offload = cpu_offload
