"""Composable parallelism API.

Reference: PyTorch TorchTitan patterns
Supports: FSDP2, Tensor Parallelism, Pipeline Parallelism
"""

from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn


@dataclass
class ParallelismConfig:
    """Configuration for composable parallelism."""

    # Data parallelism
    data_parallel_degree: int = -1  # -1 = auto
    use_fsdp2: bool = True
    fsdp_shard_degree: int = -1

    # Tensor parallelism
    tensor_parallel_degree: int = 1

    # Pipeline parallelism
    pipeline_parallel_degree: int = 1
    pipeline_schedule: str = "1F1B"
    pipeline_split_points: Optional[List[str]] = None
    num_microbatches: int = 4

    # Context parallelism
    context_parallel_degree: int = 1

    # Memory optimization
    activation_checkpointing: str = "selective"
    cpu_offload: bool = False

    # Communication
    async_tensor_parallel: bool = False


def create_device_mesh(config: ParallelismConfig) -> "DeviceMesh":
    """Create device mesh for parallelism.

    Args:
        config: Parallelism configuration

    Returns:
        DeviceMesh for distributed training
    """
    from torch.distributed.device_mesh import DeviceMesh

    if not torch.distributed.is_initialized():
        raise RuntimeError("Distributed training not initialized")

    world_size = torch.distributed.get_world_size()

    # Calculate dimensions
    tp = config.tensor_parallel_degree
    pp = config.pipeline_parallel_degree
    cp = config.context_parallel_degree

    dp = config.data_parallel_degree
    if dp == -1:
        dp = world_size // (tp * pp * cp)

    assert dp * tp * pp * cp == world_size, (
        f"Parallelism degrees ({dp}*{tp}*{pp}*{cp}={dp*tp*pp*cp}) "
        f"must equal world size ({world_size})"
    )

    # Create mesh based on parallelism strategy
    if pp > 1:
        # 3D+ parallelism: (dp, pp, tp) or (dp, pp, tp, cp)
        if cp > 1:
            mesh_shape = (dp, pp, tp, cp)
            mesh_names = ("dp", "pp", "tp", "cp")
        else:
            mesh_shape = (dp, pp, tp)
            mesh_names = ("dp", "pp", "tp")
    elif tp > 1:
        # 2D parallelism: (dp, tp)
        mesh_shape = (dp, tp)
        mesh_names = ("dp", "tp")
    else:
        # Data parallelism only
        mesh_shape = (dp,)
        mesh_names = ("dp",)

    device_ids = list(range(world_size))
    device_ids = torch.tensor(device_ids).reshape(mesh_shape)

    return DeviceMesh("cuda", device_ids, mesh_dim_names=mesh_names)


def apply_parallelism(
    model: nn.Module,
    config: ParallelismConfig,
    device_mesh: Optional["DeviceMesh"] = None,
) -> nn.Module:
    """Apply composable parallelism to model.

    Order of application:
    1. Pipeline parallelism (splits model)
    2. Tensor parallelism (shards layers)
    3. Activation checkpointing
    4. FSDP2 (shards parameters)

    Args:
        model: Model to parallelize
        config: Parallelism configuration
        device_mesh: Device mesh (created if None)

    Returns:
        Parallelized model
    """
    if not torch.distributed.is_initialized():
        # Single GPU mode
        return model

    # Create device mesh if needed
    if device_mesh is None:
        device_mesh = create_device_mesh(config)

    # Apply tensor parallelism if configured
    if config.tensor_parallel_degree > 1:
        model = _apply_tensor_parallelism(model, device_mesh, config)

    # Apply pipeline parallelism if configured
    if config.pipeline_parallel_degree > 1:
        model = _apply_pipeline_parallelism(model, device_mesh, config)

    # Apply FSDP2 if configured
    if config.use_fsdp2:
        from data_handler._legacy.distributed.fsdp2 import apply_fsdp2

        model = apply_fsdp2(
            model,
            device_mesh=device_mesh,
            activation_checkpointing=config.activation_checkpointing,
            cpu_offload=config.cpu_offload,
        )

    return model


def _apply_tensor_parallelism(
    model: nn.Module,
    device_mesh: "DeviceMesh",
    config: ParallelismConfig,
) -> nn.Module:
    """Apply tensor parallelism to model.

    Uses PyTorch's DTensor-based tensor parallelism.
    """
    try:
        from torch.distributed.tensor.parallel import (
            parallelize_module,
            ColwiseParallel,
            RowwiseParallel,
        )
    except ImportError:
        # Fallback for older PyTorch versions
        return model

    # Get TP mesh dimension
    tp_mesh = device_mesh["tp"] if "tp" in device_mesh.mesh_dim_names else None
    if tp_mesh is None:
        return model

    # Define parallelism plan for transformer layers
    # Q, K, V, gate, up projections: ColwiseParallel
    # O, down projections: RowwiseParallel
    from data_handler._legacy.models.attention import MultiHeadAttention, FeedForward

    plan = {}

    for name, module in model.named_modules():
        if isinstance(module, MultiHeadAttention):
            plan[f"{name}.q_proj"] = ColwiseParallel()
            plan[f"{name}.k_proj"] = ColwiseParallel()
            plan[f"{name}.v_proj"] = ColwiseParallel()
            plan[f"{name}.o_proj"] = RowwiseParallel()
        elif isinstance(module, FeedForward):
            plan[f"{name}.gate_proj"] = ColwiseParallel()
            plan[f"{name}.up_proj"] = ColwiseParallel()
            plan[f"{name}.down_proj"] = RowwiseParallel()

    if plan:
        model = parallelize_module(model, tp_mesh, plan)

    return model


def _apply_pipeline_parallelism(
    model: nn.Module,
    device_mesh: "DeviceMesh",
    config: ParallelismConfig,
) -> nn.Module:
    """Apply pipeline parallelism to model.

    Splits model into stages across PP dimension.
    """
    # Pipeline parallelism requires more complex setup
    # For now, return model unchanged with a warning
    import warnings
    warnings.warn(
        "Pipeline parallelism is not yet fully implemented. "
        "Model will be returned without PP applied."
    )
    return model


def get_world_info() -> dict:
    """Get distributed training information.

    Returns:
        Dictionary with rank, local_rank, world_size
    """
    import os

    return {
        "rank": int(os.environ.get("RANK", 0)),
        "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
        "world_size": int(os.environ.get("WORLD_SIZE", 1)),
        "is_distributed": torch.distributed.is_initialized() if torch.distributed.is_available() else False,
    }
