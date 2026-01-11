"""FSDP wrapping utilities for distributed training."""

import functools
from typing import Optional, Type

import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


def get_sharding_strategy(strategy: str) -> ShardingStrategy:
    """
    Get FSDP sharding strategy from string name.

    Args:
        strategy: Strategy name ("FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD")

    Returns:
        ShardingStrategy enum value
    """
    strategy_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,  # ZERO-3
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,  # ZERO-2
        "NO_SHARD": ShardingStrategy.NO_SHARD,  # DDP-like
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,  # Shard within node
    }
    if strategy not in strategy_map:
        raise ValueError(f"Unknown sharding strategy: {strategy}. Choose from {list(strategy_map.keys())}")
    return strategy_map[strategy]


def get_mixed_precision_policy(
    param_dtype: str = "bfloat16",
    reduce_dtype: str = "bfloat16",
    buffer_dtype: str = "bfloat16",
) -> MixedPrecision:
    """
    Create mixed precision policy for FSDP.

    Args:
        param_dtype: Data type for parameters
        reduce_dtype: Data type for gradient reduction
        buffer_dtype: Data type for buffers

    Returns:
        MixedPrecision configuration
    """
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    return MixedPrecision(
        param_dtype=dtype_map.get(param_dtype, torch.bfloat16),
        reduce_dtype=dtype_map.get(reduce_dtype, torch.bfloat16),
        buffer_dtype=dtype_map.get(buffer_dtype, torch.bfloat16),
    )


def wrap_model_fsdp(
    model: nn.Module,
    transformer_layer_cls: Type[nn.Module],
    sharding_strategy: str = "FULL_SHARD",
    mixed_precision: bool = True,
    param_dtype: str = "bfloat16",
    activation_checkpointing: bool = True,
    use_orig_params: bool = True,
    backward_prefetch: str = "BACKWARD_PRE",
) -> FSDP:
    """
    Wrap model with FSDP for distributed training.

    Args:
        model: The model to wrap
        transformer_layer_cls: The transformer block class for auto-wrap policy
        sharding_strategy: FSDP sharding strategy
        mixed_precision: Whether to use mixed precision
        param_dtype: Parameter data type for mixed precision
        activation_checkpointing: Whether to enable activation checkpointing
        use_orig_params: Keep original parameter references (needed for optimizer groups)
        backward_prefetch: Backward prefetch strategy (BACKWARD_PRE, BACKWARD_POST, or None)

    Returns:
        FSDP-wrapped model
    """
    from torch.distributed.fsdp import BackwardPrefetch
    import logging

    logger = logging.getLogger(__name__)

    # Auto-wrap policy: wrap at transformer layer boundaries
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={transformer_layer_cls},
    )

    # Mixed precision policy
    mp_policy = None
    if mixed_precision:
        mp_policy = get_mixed_precision_policy(
            param_dtype=param_dtype,
            reduce_dtype=param_dtype,
            buffer_dtype=param_dtype,
        )

    # Sharding strategy
    strategy = get_sharding_strategy(sharding_strategy)

    # Backward prefetch strategy
    # IMPORTANT: Disable prefetching when activation checkpointing is enabled
    # to avoid NCCL deadlocks (prefetch conflicts with AC's re-run of forward)
    prefetch = None
    if activation_checkpointing:
        logger.info("Activation checkpointing enabled - disabling backward_prefetch to avoid NCCL deadlocks")
        prefetch = None
    elif backward_prefetch and backward_prefetch.upper() != "NONE":
        prefetch_map = {
            "BACKWARD_PRE": BackwardPrefetch.BACKWARD_PRE,
            "BACKWARD_POST": BackwardPrefetch.BACKWARD_POST,
        }
        prefetch = prefetch_map.get(backward_prefetch.upper())

    # Wrap with FSDP
    wrapped_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=strategy,
        mixed_precision=mp_policy,
        device_id=torch.cuda.current_device(),
        use_orig_params=use_orig_params,
        backward_prefetch=prefetch,
    )

    # Apply activation checkpointing
    if activation_checkpointing:
        apply_activation_checkpointing(wrapped_model, transformer_layer_cls)

    return wrapped_model


def apply_activation_checkpointing(
    model: FSDP,
    transformer_layer_cls: Type[nn.Module],
) -> None:
    """
    Apply activation checkpointing to transformer layers.

    This reduces memory by recomputing activations during backward pass
    instead of storing them.

    Args:
        model: FSDP-wrapped model
        transformer_layer_cls: Transformer layer class to checkpoint
    """
    try:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            apply_activation_checkpointing as torch_apply_ac,
            checkpoint_wrapper,
        )

        check_fn = lambda submodule: isinstance(submodule, transformer_layer_cls)

        # PyTorch 2.9+ uses checkpoint_wrapper_fn instead of checkpoint_impl
        non_reentrant_wrapper = functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )

        torch_apply_ac(
            model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=check_fn,
        )
    except (ImportError, TypeError):
        # Fallback for older PyTorch versions
        from torch.utils.checkpoint import checkpoint

        def apply_checkpoint_to_layers(module: nn.Module):
            for child in module.children():
                if isinstance(child, transformer_layer_cls):
                    # Store original forward
                    original_forward = child.forward

                    def checkpointed_forward(*args, **kwargs):
                        return checkpoint(original_forward, *args, use_reentrant=False, **kwargs)

                    child.forward = checkpointed_forward
                else:
                    apply_checkpoint_to_layers(child)

        apply_checkpoint_to_layers(model)


def get_fsdp_state_dict(model: FSDP, rank: int = 0) -> dict:
    """
    Get state dict from FSDP model.

    Gathers full state dict on rank 0, returns empty dict on other ranks.

    Args:
        model: FSDP-wrapped model
        rank: Current process rank

    Returns:
        Full state dict on rank 0, empty dict otherwise
    """
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType

    full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
        state_dict = model.state_dict()

    return state_dict


def load_fsdp_state_dict(model: FSDP, state_dict: dict) -> None:
    """
    Load state dict into FSDP model.

    Args:
        model: FSDP-wrapped model
        state_dict: State dict to load
    """
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType

    full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
        model.load_state_dict(state_dict)


def setup_distributed() -> tuple[int, int, int]:
    """
    Setup distributed training environment.

    Returns:
        Tuple of (rank, local_rank, world_size)
    """
    import os

    import torch.distributed as dist

    # Check if already initialized
    if dist.is_initialized():
        return dist.get_rank(), int(os.environ.get("LOCAL_RANK", 0)), dist.get_world_size()

    # Get environment variables
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Initialize process group
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    """Cleanup distributed training environment."""
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()
